//! The interactive TUI. Main thread renders; the agent runs on its own
//! thread and streams AgentEvents over a channel. Draws only when dirty,
//! coalescing stream events per frame.

mod hl;
mod input;
mod md;
mod theme;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags,
    MouseEventKind,
};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph};
use ratatui::{Frame, Terminal, backend::CrosstermBackend};

use m_core::agent::{Agent, AgentEvent, StopReason};
use m_core::config::Config;
use m_core::provider::{Timings, Usage};
use m_core::session::Session;

use input::Editor;

// ---------------------------------------------------------------- messages

enum AgentCmd {
    Run(String),
    NewSession,
    LoadSession(PathBuf),
    Compact,
}

enum UiMsg {
    Ev(AgentEvent),
    RunDone(StopReason),
    RunErr(String),
    SessionInfo { id: String, cells: Vec<CellKind> },
}

// ---------------------------------------------------------------- cells

#[derive(Debug)]
enum CellKind {
    User(String),
    Queued(String),
    Thinking {
        text: String,
        done: bool,
        expanded: bool,
    },
    Assistant {
        md: String,
        done: bool,
    },
    Tool {
        name: String,
        summary: String,
        output: String,
        is_error: Option<bool>,
        detail: Option<String>,
        expanded: bool,
    },
    Notice(String),
    ErrorCell(String),
}

struct Cell {
    kind: CellKind,
    version: u64,
    cache: Option<(u16, u64, Vec<Line<'static>>)>,
}

impl Cell {
    fn new(kind: CellKind) -> Cell {
        Cell {
            kind,
            version: 0,
            cache: None,
        }
    }
    fn touch(&mut self) {
        self.version += 1;
    }
    fn lines(&mut self, width: u16) -> &[Line<'static>] {
        let stale = match &self.cache {
            Some((w, v, _)) => *w != width || *v != self.version,
            None => true,
        };
        if stale {
            let lines = render_cell(&self.kind, width);
            self.cache = Some((width, self.version, lines));
        }
        &self.cache.as_ref().unwrap().2
    }
}

fn render_cell(kind: &CellKind, width: u16) -> Vec<Line<'static>> {
    let w = width as usize;
    match kind {
        CellKind::User(text) => {
            let mut spans = vec![Span::styled("❯ ", theme::user_tag())];
            spans.push(Span::styled(text.clone(), Style::default().bold()));
            md::wrap_spans(spans, w, "")
        }
        CellKind::Queued(text) => md::wrap_spans(
            vec![Span::styled(format!("⧗ queued: {text}"), theme::dim())],
            w,
            "",
        ),
        CellKind::Thinking {
            text,
            done,
            expanded,
        } => {
            if *done && !*expanded {
                let words = text.split_whitespace().count();
                vec![Line::styled(
                    format!("✱ thought for {words} words (ctrl+t)"),
                    theme::thinking(),
                )]
            } else {
                let mut lines = md::wrap_spans(
                    vec![Span::styled(text.clone(), theme::thinking())],
                    w.saturating_sub(2),
                    "",
                );
                for l in &mut lines {
                    l.spans.insert(0, Span::styled("┆ ", theme::dim()));
                }
                lines
            }
        }
        CellKind::Assistant { md: text, .. } => md::render(text, width),
        CellKind::Tool {
            name,
            summary,
            output,
            is_error,
            detail,
            expanded,
        } => {
            let (glyph, gstyle) = match is_error {
                None => ("…", theme::dim()),
                Some(false) => ("✓", Style::default().fg(theme::ADD)),
                Some(true) => ("✗", theme::error()),
            };
            let mut lines = md::wrap_spans(
                vec![
                    Span::styled(format!("▸ {name} "), theme::tool_tag()),
                    Span::styled(summary.clone(), theme::dim()),
                    Span::raw(" "),
                    Span::styled(glyph.to_string(), gstyle),
                ],
                w,
                "",
            );
            let body = detail.as_deref().unwrap_or(output.as_str());
            let show_body = *expanded || (*is_error == Some(true) && detail.is_none());
            if show_body && !body.is_empty() {
                let max = if *expanded { 400 } else { 6 };
                for l in body.lines().take(max) {
                    let style = if detail.is_some() {
                        match l.as_bytes().first() {
                            Some(b'+') => Style::default().fg(theme::ADD),
                            Some(b'-') => Style::default().fg(theme::DEL),
                            _ => theme::dim(),
                        }
                    } else {
                        theme::dim()
                    };
                    let clipped: String = l.chars().take(w.saturating_sub(4)).collect();
                    lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(clipped, style),
                    ]));
                }
                let total = body.lines().count();
                if total > max {
                    lines.push(Line::styled(
                        format!("  (… {} more lines, ctrl+o …)", total - max),
                        theme::dim(),
                    ));
                }
            }
            lines
        }
        CellKind::Notice(text) => {
            md::wrap_spans(vec![Span::styled(format!("· {text}"), theme::dim())], w, "")
        }
        CellKind::ErrorCell(text) => md::wrap_spans(
            vec![Span::styled(format!("✗ {text}"), theme::error())],
            w,
            "",
        ),
    }
}

// ---------------------------------------------------------------- overlays

struct Picker {
    items: Vec<(PathBuf, u64, String)>,
    selected: usize,
}

const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/help", "show help"),
    ("/new", "start a fresh session"),
    ("/resume", "pick a previous session"),
    ("/compact", "summarize the session to free context"),
    ("/skills", "list discovered skills"),
    ("/quit", "exit m"),
];

// ---------------------------------------------------------------- app

struct Telemetry {
    prompt_tokens: u64,
    tok_per_sec: f64,
    draft_accept: Option<f64>,
    cached: u64,
}

pub struct App {
    cells: Vec<Cell>,
    editor: Editor,
    running: bool,
    scroll_up: usize,
    telemetry: Option<Telemetry>,
    ctx_limit: Arc<AtomicUsize>,
    cancel: Arc<AtomicBool>,
    steer: Arc<Mutex<std::collections::VecDeque<String>>>,
    cmd_tx: mpsc::Sender<AgentCmd>,
    ui_rx: mpsc::Receiver<UiMsg>,
    profile_label: String,
    session_id: String,
    n_skills: usize,
    user_commands: Vec<m_core::context::CommandTemplate>,
    picker: Option<Picker>,
    slash_sel: usize,
    quit_armed: Option<Instant>,
    cwd: PathBuf,
    dirty: bool,
    quit: bool,
}

pub fn run_tui(
    cfg: Config,
    cwd: PathBuf,
    resume: Option<PathBuf>,
    t0: Instant,
) -> std::io::Result<i32> {
    hl::preload();

    let (sys, skills) = crate::build_env(&cwd);
    let n_skills = skills.len();
    let user_commands = m_core::context::discover_commands(&cwd);
    let agent = match resume {
        Some(p) => Agent::resume(cfg.clone(), cwd.clone(), sys, skills, &p),
        None => Agent::new(cfg.clone(), cwd.clone(), sys, skills),
    };
    let agent = match agent {
        Ok(a) => a,
        Err(e) => {
            eprintln!("m: {e}");
            return Ok(2);
        }
    };

    let (cmd_tx, cmd_rx) = mpsc::channel::<AgentCmd>();
    let (ui_tx, ui_rx) = mpsc::channel::<UiMsg>();

    let mut app = App {
        cells: session_cells(&agent.session),
        editor: Editor::default(),
        running: false,
        scroll_up: 0,
        telemetry: None,
        ctx_limit: agent.ctx_limit_handle(),
        cancel: agent.cancel.clone(),
        steer: agent.steer.clone(),
        cmd_tx,
        ui_rx,
        profile_label: format!("{}/{}", cfg.profile_name, cfg.profile.model),
        session_id: agent.session.id.clone(),
        n_skills,
        user_commands,
        picker: None,
        slash_sel: 0,
        quit_armed: None,
        cwd: cwd.clone(),
        dirty: true,
        quit: false,
    };
    if app.cells.is_empty() {
        app.cells.push(Cell::new(CellKind::Notice(format!(
            "m · {} · {} · /help for commands",
            app.profile_label,
            cwd.display()
        ))));
    }

    spawn_agent_thread(agent, cmd_rx, ui_tx);

    // Terminal setup with restore-on-panic.
    crossterm::terminal::enable_raw_mode()?;
    let mut out = std::io::stdout();
    crossterm::execute!(
        out,
        crossterm::terminal::EnterAlternateScreen,
        event::EnableMouseCapture
    )?;
    let kitty = crossterm::terminal::supports_keyboard_enhancement().unwrap_or(false);
    if kitty {
        crossterm::execute!(
            out,
            event::PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
            )
        )?;
    }
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal(kitty);
        default_hook(info);
    }));

    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;
    let perf = std::env::var("M_PERF").is_ok();
    let mut first_frame: Option<(Duration, u64)> = None;
    if perf {
        terminal.draw(|f| app.draw(f))?;
        first_frame = Some((t0.elapsed(), rss_kb()));
    }
    let res = app.event_loop(&mut terminal);
    restore_terminal(kitty);
    if let Some((ttff, rss)) = first_frame {
        eprintln!(
            "m: first frame {:.1}ms · rss {:.1}MB",
            ttff.as_secs_f64() * 1000.0,
            rss as f64 / 1024.0
        );
    }
    res?;
    Ok(0)
}

fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

fn restore_terminal(kitty: bool) {
    let mut out = std::io::stdout();
    if kitty {
        crossterm::execute!(out, event::PopKeyboardEnhancementFlags).ok();
    }
    crossterm::execute!(
        out,
        event::DisableMouseCapture,
        crossterm::terminal::LeaveAlternateScreen
    )
    .ok();
    crossterm::terminal::disable_raw_mode().ok();
}

fn spawn_agent_thread(
    mut agent: Agent,
    cmd_rx: mpsc::Receiver<AgentCmd>,
    ui_tx: mpsc::Sender<UiMsg>,
) {
    std::thread::spawn(move || {
        while let Ok(cmd) = cmd_rx.recv() {
            match cmd {
                AgentCmd::Run(prompt) => {
                    agent.cancel.store(false, Ordering::SeqCst);
                    let tx = ui_tx.clone();
                    let mut on_event = move |ev: AgentEvent| {
                        tx.send(UiMsg::Ev(ev)).ok();
                    };
                    match agent.run_prompt(&prompt, &mut on_event) {
                        Ok(outcome) => {
                            ui_tx.send(UiMsg::RunDone(outcome.stop)).ok();
                        }
                        Err(e) => {
                            ui_tx.send(UiMsg::RunErr(e.to_string())).ok();
                        }
                    }
                }
                AgentCmd::NewSession => match agent.new_session() {
                    Ok(()) => {
                        ui_tx
                            .send(UiMsg::SessionInfo {
                                id: agent.session.id.clone(),
                                cells: vec![CellKind::Notice("new session".into())],
                            })
                            .ok();
                    }
                    Err(e) => {
                        ui_tx.send(UiMsg::RunErr(e.to_string())).ok();
                    }
                },
                AgentCmd::LoadSession(path) => match agent.load_session(&path) {
                    Ok(()) => {
                        let cells: Vec<CellKind> = kinds_of(session_cells(&agent.session));
                        ui_tx
                            .send(UiMsg::SessionInfo {
                                id: agent.session.id.clone(),
                                cells,
                            })
                            .ok();
                    }
                    Err(e) => {
                        ui_tx.send(UiMsg::RunErr(e.to_string())).ok();
                    }
                },
                AgentCmd::Compact => {
                    agent.cancel.store(false, Ordering::SeqCst);
                    let tx = ui_tx.clone();
                    let mut on_event = move |ev: AgentEvent| {
                        tx.send(UiMsg::Ev(ev)).ok();
                    };
                    match agent.compact(&mut on_event) {
                        Ok(()) => {
                            ui_tx
                                .send(UiMsg::SessionInfo {
                                    id: agent.session.id.clone(),
                                    cells: vec![CellKind::Notice(
                                        "session compacted into a fresh context".into(),
                                    )],
                                })
                                .ok();
                        }
                        Err(e) => {
                            ui_tx.send(UiMsg::RunErr(e.to_string())).ok();
                        }
                    }
                }
            }
        }
    });
}

fn kinds_of(cells: Vec<Cell>) -> Vec<CellKind> {
    cells.into_iter().map(|c| c.kind).collect()
}

/// Rebuild transcript cells from a loaded session.
fn session_cells(session: &Session) -> Vec<Cell> {
    let mut cells: Vec<Cell> = Vec::new();
    let mut open_tools: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for msg in &session.messages {
        match msg.role.as_str() {
            "user" => cells.push(Cell::new(CellKind::User(
                msg.content.clone().unwrap_or_default(),
            ))),
            "assistant" => {
                if let Some(r) = &msg.reasoning
                    && !r.is_empty()
                {
                    cells.push(Cell::new(CellKind::Thinking {
                        text: r.clone(),
                        done: true,
                        expanded: false,
                    }));
                }
                if let Some(c) = &msg.content
                    && !c.is_empty()
                {
                    cells.push(Cell::new(CellKind::Assistant {
                        md: c.clone(),
                        done: true,
                    }));
                }
                for call in msg.tool_calls.iter().flatten() {
                    cells.push(Cell::new(CellKind::Tool {
                        name: call.function.name.clone(),
                        summary: crate::summarize_args(
                            &call.function.name,
                            &call.function.arguments,
                        ),
                        output: String::new(),
                        is_error: None,
                        detail: None,
                        expanded: false,
                    }));
                    open_tools.insert(call.id.clone(), cells.len() - 1);
                }
            }
            "tool" => {
                if let Some(id) = &msg.tool_call_id
                    && let Some(&i) = open_tools.get(id)
                    && let CellKind::Tool {
                        output, is_error, ..
                    } = &mut cells[i].kind
                {
                    let content = msg.content.clone().unwrap_or_default();
                    *is_error = Some(
                        content.starts_with("Error")
                            || content.starts_with("Exit code")
                            || content.starts_with("Command timed out"),
                    );
                    *output = content;
                }
            }
            _ => {}
        }
    }
    cells
}

impl App {
    fn event_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> std::io::Result<()> {
        let frame_budget = Duration::from_millis(16);
        let mut last_draw = Instant::now() - frame_budget;
        while !self.quit {
            // Drain agent events.
            while let Ok(msg) = self.ui_rx.try_recv() {
                self.apply(msg);
            }
            // Input.
            if event::poll(Duration::from_millis(if self.running { 33 } else { 250 }))? {
                match event::read()? {
                    Event::Key(k) if k.kind != KeyEventKind::Release => self.on_key(k),
                    Event::Mouse(me) => match me.kind {
                        MouseEventKind::ScrollUp => {
                            self.scroll_up += 3;
                            self.dirty = true;
                        }
                        MouseEventKind::ScrollDown => {
                            self.scroll_up = self.scroll_up.saturating_sub(3);
                            self.dirty = true;
                        }
                        _ => {}
                    },
                    Event::Resize(..) => self.dirty = true,
                    Event::Paste(s) => {
                        self.editor.insert_str(&s);
                        self.dirty = true;
                    }
                    _ => {}
                }
            }
            // Spinner animation while running.
            if self.running && last_draw.elapsed() >= Duration::from_millis(120) {
                self.dirty = true;
            }
            if self.dirty && last_draw.elapsed() >= frame_budget {
                terminal.draw(|f| self.draw(f))?;
                last_draw = Instant::now();
                self.dirty = false;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------ events

    fn apply(&mut self, msg: UiMsg) {
        self.dirty = true;
        match msg {
            UiMsg::Ev(ev) => self.apply_agent_event(ev),
            UiMsg::RunDone(stop) => {
                self.running = false;
                self.finalize_open_cells();
                match stop {
                    StopReason::Done => {}
                    StopReason::Cancelled => self.notice("cancelled"),
                    StopReason::MaxTurns => self.notice("stopped at turn limit"),
                    StopReason::Length => self.notice("stopped: token limit"),
                }
            }
            UiMsg::RunErr(e) => {
                self.running = false;
                self.finalize_open_cells();
                self.cells.push(Cell::new(CellKind::ErrorCell(e)));
            }
            UiMsg::SessionInfo { id, cells } => {
                self.running = false;
                self.session_id = id;
                self.cells = cells.into_iter().map(Cell::new).collect();
                self.telemetry = None;
                self.scroll_up = 0;
            }
        }
    }

    fn apply_agent_event(&mut self, ev: AgentEvent) {
        match ev {
            AgentEvent::Reasoning(s) => {
                if let Some(CellKind::Thinking {
                    text, done: false, ..
                }) = self.cells.last_mut().map(|c| &mut c.kind)
                {
                    text.push_str(&s);
                } else {
                    self.cells.push(Cell::new(CellKind::Thinking {
                        text: s,
                        done: false,
                        expanded: false,
                    }));
                }
                self.cells.last_mut().unwrap().touch();
            }
            AgentEvent::Content(s) => {
                if let Some(CellKind::Assistant { md, done: false }) =
                    self.cells.last_mut().map(|c| &mut c.kind)
                {
                    md.push_str(&s);
                } else {
                    self.close_thinking();
                    self.cells
                        .push(Cell::new(CellKind::Assistant { md: s, done: false }));
                }
                self.cells.last_mut().unwrap().touch();
            }
            AgentEvent::ToolStart { name, args } => {
                self.close_thinking();
                self.finalize_open_cells();
                let summary = crate::summarize_args(&name, &args);
                self.cells.push(Cell::new(CellKind::Tool {
                    name,
                    summary,
                    output: String::new(),
                    is_error: None,
                    detail: None,
                    expanded: false,
                }));
            }
            AgentEvent::ToolEnd {
                output,
                is_error,
                detail,
                ..
            } => {
                for cell in self.cells.iter_mut().rev() {
                    if let CellKind::Tool {
                        output: o,
                        is_error: e,
                        detail: d,
                        ..
                    } = &mut cell.kind
                        && e.is_none()
                    {
                        *o = output;
                        *e = Some(is_error);
                        *d = detail;
                        cell.touch();
                        break;
                    }
                }
            }
            AgentEvent::UserInjected(text) => {
                // Promote the matching queued cell.
                for cell in self.cells.iter_mut() {
                    if matches!(&cell.kind, CellKind::Queued(t) if *t == text) {
                        cell.kind = CellKind::User(text.clone());
                        cell.touch();
                        return;
                    }
                }
                self.cells.push(Cell::new(CellKind::User(text)));
            }
            AgentEvent::Telemetry { usage, timings } => {
                self.telemetry = Some(telemetry_of(usage, timings, self.telemetry.take()));
            }
            AgentEvent::Notice(n) => self.notice(n),
        }
    }

    fn close_thinking(&mut self) {
        if let Some(cell) = self.cells.last_mut()
            && let CellKind::Thinking { done, .. } = &mut cell.kind
            && !*done
        {
            *done = true;
            cell.touch();
        }
    }

    fn finalize_open_cells(&mut self) {
        for cell in self.cells.iter_mut() {
            let touched = match &mut cell.kind {
                CellKind::Thinking { done, .. } if !*done => {
                    *done = true;
                    true
                }
                CellKind::Assistant { done, .. } if !*done => {
                    *done = true;
                    true
                }
                _ => false,
            };
            if touched {
                cell.touch();
            }
        }
    }

    fn notice(&mut self, n: impl Into<String>) {
        self.cells.push(Cell::new(CellKind::Notice(n.into())));
    }

    // ------------------------------------------------------------ keys

    fn on_key(&mut self, k: KeyEvent) {
        self.dirty = true;
        // Overlay first.
        if self.picker.is_some() {
            self.picker_key(k);
            return;
        }
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let alt = k.modifiers.contains(KeyModifiers::ALT);
        let shift = k.modifiers.contains(KeyModifiers::SHIFT);
        match k.code {
            KeyCode::Char('c') if ctrl => {
                if self.running {
                    self.cancel.store(true, Ordering::SeqCst);
                } else if !self.editor.is_empty() {
                    self.editor.take();
                } else if let Some(t) = self.quit_armed
                    && t.elapsed() < Duration::from_millis(1500)
                {
                    self.quit = true;
                } else {
                    self.quit_armed = Some(Instant::now());
                    self.notice("press ctrl+c again to quit");
                }
            }
            KeyCode::Char('d') if ctrl && self.editor.is_empty() => self.quit = true,
            KeyCode::Esc => {
                if self.running {
                    self.cancel.store(true, Ordering::SeqCst);
                } else if !self.editor.is_empty() {
                    self.editor.take();
                }
            }
            KeyCode::Enter if shift || alt || ctrl => self.editor.insert('\n'),
            KeyCode::Char('j') if ctrl => self.editor.insert('\n'),
            KeyCode::Enter => self.submit(),
            KeyCode::Char('t') if ctrl => {
                self.toggle_last(|k| matches!(k, CellKind::Thinking { .. }))
            }
            KeyCode::Char('o') if ctrl => self.toggle_last(|k| matches!(k, CellKind::Tool { .. })),
            KeyCode::Char('r') if ctrl => self.open_picker(),
            KeyCode::Char('l') if ctrl => {
                self.cells
                    .retain(|c| !matches!(c.kind, CellKind::Notice(_)));
            }
            KeyCode::Char('u') if ctrl => self.editor.kill_to_start(),
            KeyCode::Char('k') if ctrl => self.editor.kill_to_end(),
            KeyCode::Char('y') if ctrl => self.editor.yank(),
            KeyCode::Char('w') if ctrl => self.editor.delete_word_back(),
            KeyCode::Char('a') if ctrl => self.editor.home(),
            KeyCode::Char('e') if ctrl => self.editor.end(),
            KeyCode::Backspace => self.editor.backspace(),
            KeyCode::Delete => self.editor.delete(),
            KeyCode::Left if alt || ctrl => self.editor.word_left(),
            KeyCode::Right if alt || ctrl => self.editor.word_right(),
            KeyCode::Left => self.editor.left(),
            KeyCode::Right => self.editor.right(),
            KeyCode::Home => self.editor.home(),
            KeyCode::End => self.editor.end(),
            KeyCode::Up => {
                if self.slash_active() {
                    self.slash_sel = self.slash_sel.saturating_sub(1);
                } else {
                    self.editor.up();
                }
            }
            KeyCode::Down => {
                if self.slash_active() {
                    self.slash_sel += 1;
                } else {
                    self.editor.down();
                }
            }
            KeyCode::PageUp => self.scroll_up += 10,
            KeyCode::PageDown => self.scroll_up = self.scroll_up.saturating_sub(10),
            KeyCode::Tab if self.slash_active() => self.slash_complete(),
            KeyCode::Char(c) if !ctrl && !alt => {
                self.editor.insert(c);
                if c == '/' || self.slash_active() {
                    self.slash_sel = 0;
                }
            }
            _ => {}
        }
    }

    fn toggle_last(&mut self, pred: impl Fn(&CellKind) -> bool) {
        for cell in self.cells.iter_mut().rev() {
            if pred(&cell.kind) {
                match &mut cell.kind {
                    CellKind::Thinking { expanded, .. } | CellKind::Tool { expanded, .. } => {
                        *expanded = !*expanded;
                        cell.touch();
                    }
                    _ => {}
                }
                return;
            }
        }
    }

    fn slash_active(&self) -> bool {
        let t = self.editor.text();
        t.starts_with('/') && !t.contains(' ') && !t.contains('\n')
    }

    fn slash_matches(&self) -> Vec<(String, String)> {
        let t = self.editor.text();
        let mut out: Vec<(String, String)> = SLASH_COMMANDS
            .iter()
            .filter(|(c, _)| c.starts_with(t))
            .map(|(c, d)| (c.to_string(), d.to_string()))
            .collect();
        for cmd in &self.user_commands {
            let slash = format!("/{}", cmd.name);
            if slash.starts_with(t) {
                out.push((slash, cmd.description.clone()));
            }
        }
        out
    }

    fn slash_complete(&mut self) {
        let matches = self.slash_matches();
        if let Some((cmd, _)) = matches.get(self.slash_sel.min(matches.len().saturating_sub(1))) {
            self.editor.set(cmd);
        }
    }

    fn submit(&mut self) {
        if self.editor.is_empty() {
            return;
        }
        let t = self.editor.text();
        if t.starts_with('/') && !t.contains('\n') {
            if self.slash_active() {
                self.slash_complete();
            }
            let cmd = self.editor.take();
            self.run_slash(&cmd);
            return;
        }
        let text = self.editor.take();
        if text.trim().is_empty() {
            return;
        }
        self.scroll_up = 0;
        if self.running {
            self.steer.lock().unwrap().push_back(text.clone());
            self.cells.push(Cell::new(CellKind::Queued(text)));
        } else {
            self.cells.push(Cell::new(CellKind::User(text.clone())));
            self.running = true;
            self.cmd_tx.send(AgentCmd::Run(text)).ok();
        }
    }

    fn run_slash(&mut self, cmd: &str) {
        let (head, args) = match cmd.trim().split_once(' ') {
            Some((h, a)) => (h, a.trim()),
            None => (cmd.trim(), ""),
        };
        // User-defined markdown templates.
        let tpl = self
            .user_commands
            .iter()
            .find(|t| format!("/{}", t.name) == head)
            .cloned();
        if let Some(t) = tpl {
            match m_core::context::expand_command(&t.path, args) {
                Ok(prompt) => {
                    if self.running {
                        self.steer.lock().unwrap().push_back(prompt.clone());
                        self.cells.push(Cell::new(CellKind::Queued(prompt)));
                    } else {
                        self.cells
                            .push(Cell::new(CellKind::User(cmd.trim().to_string())));
                        self.running = true;
                        self.cmd_tx.send(AgentCmd::Run(prompt)).ok();
                    }
                }
                Err(e) => self.notice(format!("command template: {e}")),
            }
            return;
        }
        match head {
            "/help" => {
                self.notice(
                    "enter send · shift/alt+enter newline · esc cancel · ctrl+c ×2 quit · \
                     ctrl+o expand tool · ctrl+t expand thinking · ctrl+r sessions · \
                     pgup/pgdn or wheel scroll · /new /resume /compact /quit",
                );
            }
            "/quit" => self.quit = true,
            "/new" => {
                if self.running {
                    self.notice("busy — esc to cancel first");
                } else {
                    self.cmd_tx.send(AgentCmd::NewSession).ok();
                }
            }
            "/compact" => {
                if self.running {
                    self.notice("busy — esc to cancel first");
                } else {
                    self.running = true;
                    self.notice("compacting…");
                    self.cmd_tx.send(AgentCmd::Compact).ok();
                }
            }
            "/resume" => self.open_picker(),
            "/skills" => {
                let (sys_skills, cmds) = (self.n_skills, self.user_commands.len());
                self.notice(format!(
                    "{sys_skills} skills discovered (see system prompt index); \
                     {cmds} user slash commands"
                ));
            }
            other => self.notice(format!("unknown command: {other}")),
        }
    }

    fn open_picker(&mut self) {
        if self.running {
            self.notice("busy — esc to cancel first");
            return;
        }
        let items = Session::list(&self.cwd);
        if items.is_empty() {
            self.notice("no sessions for this directory");
            return;
        }
        self.picker = Some(Picker { items, selected: 0 });
    }

    fn picker_key(&mut self, k: KeyEvent) {
        let Some(p) = &mut self.picker else { return };
        match k.code {
            KeyCode::Esc => self.picker = None,
            KeyCode::Up => p.selected = p.selected.saturating_sub(1),
            KeyCode::Down => p.selected = (p.selected + 1).min(p.items.len() - 1),
            KeyCode::Enter => {
                let path = p.items[p.selected].0.clone();
                self.picker = None;
                self.cmd_tx.send(AgentCmd::LoadSession(path)).ok();
            }
            _ => {}
        }
    }

    // ------------------------------------------------------------ draw

    fn draw(&mut self, f: &mut Frame) {
        let input_lines = self.editor.lines().len().clamp(1, 8) as u16;
        let [transcript_area, input_area, status_area] = Layout::vertical([
            Constraint::Min(3),
            Constraint::Length(input_lines + 2),
            Constraint::Length(1),
        ])
        .areas(f.area());

        self.draw_transcript(f, transcript_area);
        self.draw_input(f, input_area);
        self.draw_status(f, status_area);
        if self.slash_active() {
            self.draw_slash_menu(f, input_area);
        }
        if self.picker.is_some() {
            self.draw_picker(f);
        }
    }

    fn draw_transcript(&mut self, f: &mut Frame, area: Rect) {
        let width = area.width.saturating_sub(1);
        // Ensure caches, count lines.
        let mut total = 0usize;
        for cell in self.cells.iter_mut() {
            total += cell.lines(width).len() + 1; // +1 blank separator
        }
        total = total.saturating_sub(1);
        let h = area.height as usize;
        let max_scroll = total.saturating_sub(h);
        self.scroll_up = self.scroll_up.min(max_scroll);
        let start = max_scroll - self.scroll_up;

        let buf = f.buffer_mut();
        let mut row = 0usize; // global line index
        let mut y = 0u16;
        'outer: for cell in self.cells.iter_mut() {
            let lines = cell.lines(width);
            let n = lines.len() + 1;
            if row + n <= start {
                row += n;
                continue;
            }
            for line in lines {
                if row >= start {
                    if y >= area.height {
                        break 'outer;
                    }
                    buf.set_line(area.x, area.y + y, line, width);
                    y += 1;
                }
                row += 1;
            }
            row += 1; // separator
            if row > start && y < area.height {
                y += 1;
            }
        }
        if self.scroll_up > 0 {
            let tag = format!(" ↓ {} lines below ", self.scroll_up);
            let x = area.x + area.width.saturating_sub(tag.len() as u16 + 1);
            buf.set_string(x, area.y + area.height - 1, tag, theme::accent().reversed());
        }
    }

    fn draw_input(&mut self, f: &mut Frame, area: Rect) {
        let border_style = if self.running {
            theme::dim()
        } else {
            theme::accent()
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(border_style);
        let inner = block.inner(area);
        f.render_widget(block, area);
        let text: Vec<Line> = self
            .editor
            .lines()
            .iter()
            .map(|l| Line::raw(l.to_string()))
            .collect();
        let nlines = text.len() as u16;
        let scroll = nlines.saturating_sub(inner.height);
        f.render_widget(Paragraph::new(text).scroll((scroll, 0)), inner);
        let (r, c) = self.editor.cursor_rc();
        let cy = (r as u16).saturating_sub(scroll);
        if cy < inner.height {
            f.set_cursor_position((
                inner.x + (c as u16).min(inner.width.saturating_sub(1)),
                inner.y + cy,
            ));
        }
    }

    fn draw_status(&mut self, f: &mut Frame, area: Rect) {
        let mut left: Vec<Span> = vec![
            Span::styled(" m ", theme::accent().bold()),
            Span::styled(self.profile_label.clone(), theme::dim()),
        ];
        if self.running {
            const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let tick = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() / 120)
                .unwrap_or(0) as usize;
            left.push(Span::raw("  "));
            left.push(Span::styled(
                format!("{} working · esc to cancel", FRAMES[tick % FRAMES.len()]),
                theme::accent(),
            ));
        }
        let mut right_parts: Vec<String> = Vec::new();
        if let Some(t) = &self.telemetry {
            let ctx = self.ctx_limit.load(Ordering::Relaxed);
            if ctx > 0 && t.prompt_tokens > 0 {
                right_parts.push(format!("ctx {}%", t.prompt_tokens * 100 / ctx as u64));
            }
            if t.tok_per_sec > 0.0 {
                right_parts.push(format!("{:.0} tok/s", t.tok_per_sec));
            }
            if let Some(a) = t.draft_accept {
                right_parts.push(format!("mtp {:.0}%", a * 100.0));
            }
            if t.cached > 0 {
                right_parts.push(format!("cache {}", t.cached));
            }
        }
        let right = right_parts.join(" · ") + " ";
        let line = Line::from(left);
        f.render_widget(Paragraph::new(line), area);
        let rw = right.len() as u16;
        if rw < area.width {
            let rect = Rect {
                x: area.x + area.width - rw,
                width: rw,
                ..area
            };
            f.render_widget(Paragraph::new(Span::styled(right, theme::dim())), rect);
        }
    }

    fn draw_slash_menu(&mut self, f: &mut Frame, input_area: Rect) {
        let matches = self.slash_matches();
        if matches.is_empty() {
            return;
        }
        self.slash_sel = self.slash_sel.min(matches.len() - 1);
        let h = matches.len() as u16 + 2;
        let area = Rect {
            x: input_area.x + 2,
            y: input_area.y.saturating_sub(h),
            width: 44.min(f.area().width.saturating_sub(4)),
            height: h,
        };
        let lines: Vec<Line> = matches
            .iter()
            .enumerate()
            .map(|(i, (cmd, desc))| {
                let style = if i == self.slash_sel {
                    theme::accent().reversed()
                } else {
                    Style::default()
                };
                Line::from(vec![
                    Span::styled(format!(" {cmd:<10}"), style),
                    Span::styled(format!(" {desc}"), theme::dim()),
                ])
            })
            .collect();
        f.render_widget(Clear, area);
        f.render_widget(
            Paragraph::new(lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(theme::dim()),
            ),
            area,
        );
    }

    fn draw_picker(&mut self, f: &mut Frame) {
        let Some(p) = &self.picker else { return };
        let w = (f.area().width * 3 / 4).clamp(30, 100);
        let h = (p.items.len() as u16 + 2).min(f.area().height / 2).max(5);
        let area = Rect {
            x: (f.area().width - w) / 2,
            y: (f.area().height - h) / 3,
            width: w,
            height: h,
        };
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let lines: Vec<Line> = p
            .items
            .iter()
            .enumerate()
            .map(|(i, (_, created, first))| {
                let style = if i == p.selected {
                    theme::accent().reversed()
                } else {
                    Style::default()
                };
                Line::from(vec![
                    Span::styled(
                        format!(" {:>8} ", ago(now.saturating_sub(*created))),
                        theme::dim(),
                    ),
                    Span::styled(first.clone(), style),
                ])
            })
            .collect();
        f.render_widget(Clear, area);
        f.render_widget(
            Paragraph::new(lines).block(
                Block::default()
                    .title(" resume session (enter/esc) ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(theme::accent()),
            ),
            area,
        );
    }
}

fn ago(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs}s ago"),
        60..=3599 => format!("{}m ago", secs / 60),
        3600..=86399 => format!("{}h ago", secs / 3600),
        _ => format!("{}d ago", secs / 86400),
    }
}

fn telemetry_of(
    usage: Option<Usage>,
    timings: Option<Timings>,
    prev: Option<Telemetry>,
) -> Telemetry {
    let mut t = prev.unwrap_or(Telemetry {
        prompt_tokens: 0,
        tok_per_sec: 0.0,
        draft_accept: None,
        cached: 0,
    });
    if let Some(u) = usage {
        t.prompt_tokens = u.prompt_tokens + u.completion_tokens;
    }
    if let Some(ti) = timings {
        if ti.predicted_per_second > 0.0 {
            t.tok_per_sec = ti.predicted_per_second;
        }
        if ti.draft_n > 0 {
            t.draft_accept = Some(ti.draft_n_accepted as f64 / ti.draft_n as f64);
        }
        if ti.cache_n > 0 {
            t.cached = ti.cache_n;
        }
    }
    t
}
