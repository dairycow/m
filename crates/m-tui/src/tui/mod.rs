//! The interactive TUI. Main thread renders; the agent runs on its own
//! thread and streams AgentEvents over a channel. Draws only when dirty,
//! coalescing stream events per frame.

mod files;
mod fuzzy;
mod hl;
mod input;
mod md;
mod theme;

use std::io::Write;
use std::path::{Path, PathBuf};
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
    SwitchProfile(String),
}

enum UiMsg {
    Ev(AgentEvent),
    RunDone(StopReason),
    RunErr(String),
    SessionInfo {
        id: String,
        path: PathBuf,
        cells: Vec<CellKind>,
    },
    RebuildDone(Result<(), String>),
    ProfileSwitched {
        name: String,
        model: String,
    },
    AtFiles(Vec<String>),
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
    (
        "/model",
        "list or switch provider/model (e.g. zai-coding-plan/glm-5.2)",
    ),
    (
        "/reload",
        "hot-reload the running binary, keeping this session",
    ),
    (
        "/rebuild",
        "cargo build+test --release in the background, then hot-reload",
    ),
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
    /// A second sender handle so background jobs we spawn ourselves (e.g.
    /// `/rebuild`) can report back without going through the agent thread.
    ui_tx: mpsc::Sender<UiMsg>,
    profile_label: String,
    profile_name: String,
    session_id: String,
    session_path: PathBuf,
    n_skills: usize,
    user_commands: Vec<m_core::context::CommandTemplate>,
    picker: Option<Picker>,
    slash_sel: usize,
    quit_armed: Option<Instant>,
    cwd: PathBuf,
    dirty: bool,
    quit: bool,
    /// Set right after `ctrl+x` while we wait for the completing `e`.
    leader_x: Option<Instant>,
    /// Set by `on_key` when the `ctrl+x ctrl+e` sequence completes; the
    /// event loop notices this and suspends the TUI to run `$EDITOR`.
    want_editor: bool,
    /// A background `/rebuild` finished; swap into the new binary as soon
    /// as the agent isn't mid-turn.
    reload_pending: bool,
    /// Cached project file listing for the `@` picker, refreshed on every
    /// fresh `@` keystroke.
    at_files: Arc<[String]>,
    /// A listing thread is in flight (drives the "loading files…" popup).
    at_loading: bool,
    at_sel: usize,
    /// Byte offset of an `@` the user Esc'd; keeps that mention's popup
    /// closed until it's retyped.
    at_dismissed_at: Option<usize>,
    /// Kitty keyboard-enhancement protocol active (for restore /reload).
    kitty: bool,
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
    let app_ui_tx = ui_tx.clone();

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
        ui_tx: app_ui_tx,
        profile_label: format!("{}/{}", cfg.profile_name, cfg.profile.model),
        profile_name: cfg.profile_name.clone(),
        session_id: agent.session.id.clone(),
        session_path: agent.session.path.clone(),
        n_skills,
        user_commands,
        picker: None,
        slash_sel: 0,
        quit_armed: None,
        cwd: cwd.clone(),
        dirty: true,
        quit: false,
        leader_x: None,
        want_editor: false,
        reload_pending: false,
        at_files: Arc::from([]),
        at_loading: false,
        at_sel: 0,
        at_dismissed_at: None,
        kitty: false, // set once enhancement probe completes
    };
    if app.cells.is_empty() {
        app.cells.push(Cell::new(CellKind::Notice(format!(
            "m · {} · {} · /help for commands",
            app.profile_label,
            cwd.display()
        ))));
    }

    spawn_agent_thread(agent, cmd_rx, ui_tx);

    // Snapshot cooked termios *before* raw mode so signal handlers can put
    // the shell back even when Drop/panic hooks don't run (SIGTERM, etc.).
    // Without this, a killed TUI leaves the tty with ECHO/ICANON off — the
    // classic "commands work but typed text is invisible" failure mode.
    save_cooked_termios();

    // Terminal setup. `TerminalGuard` restores on every exit path (Ok, Err,
    // panic); signal handlers cover hard kills that skip unwinding.
    //
    // Install emergency signal handlers *before* the kitty probe: that
    // query blocks up to ~2s, and a SIGTERM in the window used to leave
    // the shell raw with no restore path.
    crossterm::terminal::enable_raw_mode()?;
    install_emergency_signal_handlers();
    mark_tui_active(false);
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
    mark_tui_active(kitty);
    install_panic_hook(kitty);
    app.kitty = kitty;
    let mut guard = TerminalGuard { kitty, armed: true };

    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;
    // A fresh Terminal's diff buffer starts blank, so it may skip cells it
    // thinks are already empty. After a `/reload` exec() the physical
    // screen can still hold the old process's leftover frame — force a
    // full repaint so the first draw doesn't leave stale content behind.
    terminal.clear()?;
    let perf = std::env::var("M_PERF").is_ok();
    let mut first_frame: Option<(Duration, u64)> = None;
    if perf {
        terminal.draw(|f| app.draw(f))?;
        first_frame = Some((t0.elapsed(), rss_kb()));
    }
    let res = app.event_loop(&mut terminal, kitty);
    guard.restore();
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

// ---------------------------------------------------------------- terminal lifecycle
//
// Leaving raw mode / alternate screen is the difference between a usable
// shell and one that accepts keystrokes but paints nothing (ECHO off).
// Restore is therefore best-effort, idempotent, and reachable from Drop,
// the panic hook, signal handlers, and the `/reload` exec path.

/// TUI currently owns the tty (raw + alt screen). Cleared by restore.
static TUI_ACTIVE: AtomicBool = AtomicBool::new(false);
/// Kitty keyboard progressive-enhancement stack was pushed.
static KITTY_PUSHED: AtomicBool = AtomicBool::new(false);
/// `SAVED_TERMIOS` holds a pre-raw snapshot suitable for signal-safe restore.
static TERMIOS_SAVED: AtomicBool = AtomicBool::new(false);

// Written once on the main thread before signals are installed; read from
// signal handlers afterwards. Access is synchronized by TERMIOS_SAVED + the
// single-threaded install-before-use pattern.
static mut SAVED_TERMIOS: libc::termios = unsafe { std::mem::zeroed() };

fn save_cooked_termios() {
    unsafe {
        let mut t: libc::termios = std::mem::zeroed();
        // stdin is the controlling tty in interactive use; match crossterm.
        if libc::tcgetattr(libc::STDIN_FILENO, &mut t) == 0 {
            // If we were re-exec'd without a prior restore (or started under
            // another raw-mode program), the snapshot is already raw. Treat
            // that as unusable and force a sane cooked baseline so quit can
            // never "restore" to invisible-echo mode.
            ensure_cooked_flags(&mut t);
            SAVED_TERMIOS = t;
            TERMIOS_SAVED.store(true, Ordering::SeqCst);
        }
    }
}

/// Turn on the line-discipline flags an interactive shell needs. Idempotent
/// on an already-cooked termios; repairs a raw one into something usable.
fn ensure_cooked_flags(t: &mut libc::termios) {
    // Input: map CR→NL, allow XON/XOFF; drop raw-style stripping.
    t.c_iflag |= libc::BRKINT | libc::ICRNL | libc::IMAXBEL | libc::IXON;
    t.c_iflag &= !(libc::IGNBRK | libc::INLCR | libc::IGNCR | libc::IXOFF);
    // Output: post-process NL→CRNL so shell output is visible line-by-line.
    t.c_oflag |= libc::OPOST | libc::ONLCR;
    // Local: canonical line editing + echo + signals. This is the whole fix
    // for "commands work but typed text is invisible".
    t.c_lflag |= libc::ISIG | libc::ICANON | libc::IEXTEN | libc::ECHO | libc::ECHOE | libc::ECHOK;
    t.c_lflag &= !(libc::ECHONL);
    // Non-canonical VMIN/VTIME are irrelevant once ICANON is on, but leave
    // cc[] as whatever the host had so erase/kill chars stay familiar.
}

fn mark_tui_active(kitty: bool) {
    TUI_ACTIVE.store(true, Ordering::SeqCst);
    KITTY_PUSHED.store(kitty, Ordering::SeqCst);
}

/// RAII: always put the terminal back when the TUI scope ends.
struct TerminalGuard {
    kitty: bool,
    armed: bool,
}

impl TerminalGuard {
    /// Restore now and disarm so Drop is a no-op (e.g. before `exec`).
    fn restore(&mut self) {
        if self.armed {
            restore_terminal(self.kitty);
            self.armed = false;
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if self.armed {
            restore_terminal(self.kitty);
            self.armed = false;
        }
    }
}

fn install_panic_hook(kitty: bool) {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal(kitty);
        default_hook(info);
    }));
}

fn install_emergency_signal_handlers() {
    // SIGINT is normally masked by raw mode (ISIG off) for in-band ctrl+c
    // handling, but external `kill -INT` / SIGHUP / SIGTERM / SIGQUIT still
    // arrive and would otherwise exit without restoring the tty.
    unsafe {
        let f: extern "C" fn(libc::c_int) = emergency_terminal_restore;
        let handler = f as usize;
        libc::signal(libc::SIGTERM, handler);
        libc::signal(libc::SIGHUP, handler);
        libc::signal(libc::SIGINT, handler);
        libc::signal(libc::SIGQUIT, handler);
    }
}

/// Async-signal-safe-ish emergency restore. Uses only `write` + `tcsetattr`
/// + `_exit` — not crossterm (locks / alloc).
extern "C" fn emergency_terminal_restore(sig: libc::c_int) {
    // Always try; TUI_ACTIVE just avoids a second pass if Drop already ran.
    TUI_ACTIVE.store(false, Ordering::SeqCst);
    // CSI sequences: pop kitty kbd, disable mouse (all modes crossterm enables),
    // leave alt screen, show cursor, reset SGR, reset scroll region.
    // PopKeyboardEnhancementFlags == CSI < 1 u
    const SEQ: &[u8] = b"\x1b[<1u\x1b[?1006l\x1b[?1015l\x1b[?1003l\x1b[?1002l\x1b[?1000l\x1b[?1049l\x1b[?25h\x1b[0m\x1b[r";
    unsafe {
        libc::write(libc::STDOUT_FILENO, SEQ.as_ptr() as *const _, SEQ.len());
        // Background jobs get SIGTTOU on tcsetattr to the controlling tty;
        // ignore it so the restore still lands.
        libc::signal(libc::SIGTTOU, libc::SIG_IGN);
        // Prefer /dev/tty when present: stdin may be a pipe in odd launchers,
        // while the line discipline we raw'd lives on the controlling tty.
        let mut fd = libc::STDIN_FILENO;
        let tty = libc::open(c"/dev/tty".as_ptr(), libc::O_RDWR | libc::O_NOCTTY);
        if tty >= 0 {
            fd = tty;
        }
        if TERMIOS_SAVED.load(Ordering::SeqCst) {
            let rc = libc::tcsetattr(fd, libc::TCSANOW, std::ptr::addr_of!(SAVED_TERMIOS));
            if rc != 0 {
                // Snapshot missing or rejected — synthesize cooked flags on the live attrs.
                let mut t: libc::termios = std::mem::zeroed();
                if libc::tcgetattr(fd, &mut t) == 0 {
                    // Inline the critical bits (no call into non-async-safe code).
                    t.c_iflag |= libc::BRKINT | libc::ICRNL | libc::IMAXBEL | libc::IXON;
                    t.c_oflag |= libc::OPOST | libc::ONLCR;
                    t.c_lflag |= libc::ISIG
                        | libc::ICANON
                        | libc::IEXTEN
                        | libc::ECHO
                        | libc::ECHOE
                        | libc::ECHOK;
                    libc::tcsetattr(fd, libc::TCSANOW, &t);
                }
            }
        } else {
            let mut t: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(fd, &mut t) == 0 {
                t.c_iflag |= libc::BRKINT | libc::ICRNL | libc::IMAXBEL | libc::IXON;
                t.c_oflag |= libc::OPOST | libc::ONLCR;
                t.c_lflag |= libc::ISIG
                    | libc::ICANON
                    | libc::IEXTEN
                    | libc::ECHO
                    | libc::ECHOE
                    | libc::ECHOK;
                libc::tcsetattr(fd, libc::TCSANOW, &t);
            }
        }
        if tty >= 0 {
            libc::close(tty);
        }
        libc::_exit(128 + sig);
    }
}

fn restore_terminal(kitty: bool) {
    TUI_ACTIVE.store(false, Ordering::SeqCst);
    let mut out = std::io::stdout();
    let pop_kitty = kitty || KITTY_PUSHED.load(Ordering::SeqCst);
    if pop_kitty {
        crossterm::execute!(out, event::PopKeyboardEnhancementFlags).ok();
        KITTY_PUSHED.store(false, Ordering::SeqCst);
    }
    // Show cursor + reset colors: ratatui may have left the cursor hidden or
    // an SGR attribute active if we tear down mid-frame.
    crossterm::execute!(
        out,
        event::DisableMouseCapture,
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::cursor::Show,
        crossterm::style::ResetColor,
    )
    .ok();
    // Reset scrolling region (DECSTBM) in case a partial draw left it set.
    let _ = out.write_all(b"\x1b[r");
    let _ = out.flush();
    crossterm::terminal::disable_raw_mode().ok();
    // Final authority: apply our pre-raw snapshot (with cooked flags forced
    // on). Crossterm alone is not enough after `/reload` exec, because the
    // replacement process would have saved raw mode as its "original".
    force_cooked_termios();
}

fn force_cooked_termios() {
    unsafe {
        libc::signal(libc::SIGTTOU, libc::SIG_IGN);
        // Prefer the controlling tty over stdin: they diverge when stdin is
        // redirected, and background restores need /dev/tty to avoid SIGTTOU
        // weirdness with the process-group line discipline.
        let tty = libc::open(c"/dev/tty".as_ptr(), libc::O_RDWR | libc::O_NOCTTY);
        let fd = if tty >= 0 { tty } else { libc::STDIN_FILENO };

        let ok = if TERMIOS_SAVED.load(Ordering::SeqCst) {
            libc::tcsetattr(fd, libc::TCSANOW, std::ptr::addr_of!(SAVED_TERMIOS)) == 0
        } else {
            false
        };
        if !ok {
            let mut t: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(fd, &mut t) == 0 {
                ensure_cooked_flags(&mut t);
                let _ = libc::tcsetattr(fd, libc::TCSANOW, &t);
            }
        }
        if tty >= 0 {
            libc::close(tty);
        }
    }
}

/// Re-enter the TUI's raw/alternate-screen mode after [`restore_terminal`]
/// suspended it for a child process (e.g. `$EDITOR`).
fn enter_terminal(kitty: bool) -> std::io::Result<()> {
    crossterm::terminal::enable_raw_mode()?;
    let mut out = std::io::stdout();
    crossterm::execute!(
        out,
        crossterm::terminal::EnterAlternateScreen,
        event::EnableMouseCapture
    )?;
    if kitty {
        crossterm::execute!(
            out,
            event::PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
            )
        )?;
    }
    mark_tui_active(kitty);
    Ok(())
}

/// Where `cargo build --release` puts this binary in the workspace this
/// copy of `m` was built from, derived from `CARGO_MANIFEST_DIR`
/// (`.../crates/m-tui`) rather than any runtime lookup.
fn release_binary_path() -> Option<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent()?.parent()?;
    Some(workspace_root.join("target/release/m"))
}

/// Run `cargo <args>` in `cwd`, capturing rather than inheriting its
/// stdout/stderr (the caller is a background thread while the TUI owns the
/// terminal). On failure, `label` names the step in the error message.
fn run_cargo_step(cwd: &Path, args: &[&str], label: &str) -> Result<(), String> {
    let result = std::process::Command::new("cargo")
        .args(args)
        .current_dir(cwd)
        .stdout(std::process::Stdio::null())
        .output();
    match result {
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => {
            let last_line = String::from_utf8_lossy(&out.stderr)
                .lines()
                .next_back()
                .unwrap_or("no output")
                .to_string();
            Err(format!("{label} failed: {last_line}"))
        }
        Err(e) => Err(format!("failed to run cargo {label}: {e}")),
    }
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
                                path: agent.session.path.clone(),
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
                                path: agent.session.path.clone(),
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
                                    path: agent.session.path.clone(),
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
                AgentCmd::SwitchProfile(name) => match agent.switch_profile(&name) {
                    Ok(()) => {
                        ui_tx
                            .send(UiMsg::ProfileSwitched {
                                name: agent.config.profile_name.clone(),
                                model: agent.config.profile.model.clone(),
                            })
                            .ok();
                    }
                    Err(e) => {
                        ui_tx.send(UiMsg::RunErr(e.to_string())).ok();
                    }
                },
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
        kitty: bool,
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
                    Event::Key(k) if k.kind != KeyEventKind::Release => {
                        self.on_key(k);
                        if self.want_editor {
                            self.want_editor = false;
                            self.open_external_editor(terminal, kitty);
                        }
                    }
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
            // Clear a stale "ctrl+x" leader hint nobody completed.
            if let Some(t) = self.leader_x
                && t.elapsed() >= Duration::from_millis(1500)
            {
                self.leader_x = None;
                self.dirty = true;
            }
            // A background rebuild finished; hot-reload as soon as we're idle.
            if self.reload_pending && !self.running {
                self.reload_pending = false;
                self.reload();
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
            UiMsg::SessionInfo { id, path, cells } => {
                self.running = false;
                self.session_id = id;
                self.session_path = path;
                self.cells = cells.into_iter().map(Cell::new).collect();
                self.telemetry = None;
                self.scroll_up = 0;
            }
            UiMsg::RebuildDone(Ok(())) => {
                self.notice("rebuild done — reloading…");
                self.reload_pending = true;
            }
            UiMsg::RebuildDone(Err(e)) => self.notice(format!("rebuild failed: {e}")),
            UiMsg::ProfileSwitched { name, model } => {
                let label = format!("{name}/{model}");
                if self.profile_label == label {
                    self.notice(format!("already on {label}"));
                } else {
                    self.profile_label = label.clone();
                    self.profile_name = name;
                    self.notice(format!("switched to {label}"));
                }
            }
            UiMsg::AtFiles(files) => {
                self.at_files = files.into();
                self.at_loading = false;
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

        let leader_pending = match self.leader_x.take() {
            Some(t) => t.elapsed() < Duration::from_millis(1500),
            None => false,
        };
        match leader_step(leader_pending, ctrl, k.code) {
            LeaderOutcome::Armed => {
                self.leader_x = Some(Instant::now());
                return;
            }
            LeaderOutcome::OpenEditor => {
                self.want_editor = true;
                return;
            }
            LeaderOutcome::Pass => {}
        }

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
            KeyCode::Esc if self.at_open() => {
                self.at_dismissed_at = self.editor.mention().map(|(start, _)| start);
            }
            KeyCode::Esc => {
                if self.running {
                    self.cancel.store(true, Ordering::SeqCst);
                } else if !self.editor.is_empty() {
                    self.editor.take();
                }
            }
            KeyCode::Enter if shift || alt || ctrl => self.editor.insert('\n'),
            KeyCode::Enter if self.editor.mention().is_some() => self.at_complete(),
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
            // Jump the transcript to the very top/bottom (ctrl+a/ctrl+e
            // above cover start/end of the current input line).
            KeyCode::Home => self.scroll_up = usize::MAX,
            KeyCode::End => self.scroll_up = 0,
            KeyCode::Up => {
                if self.slash_active() {
                    self.slash_sel = self.slash_sel.saturating_sub(1);
                } else if self.editor.mention().is_some() {
                    self.at_sel = self.at_sel.saturating_sub(1);
                } else {
                    self.editor.up();
                }
            }
            KeyCode::Down => {
                if self.slash_active() {
                    self.slash_sel += 1;
                } else if self.editor.mention().is_some() {
                    self.at_sel += 1;
                } else {
                    self.editor.down();
                }
            }
            KeyCode::PageUp => self.scroll_up += 10,
            KeyCode::PageDown => self.scroll_up = self.scroll_up.saturating_sub(10),
            KeyCode::Tab if self.slash_active() => self.slash_complete(),
            KeyCode::Tab if self.editor.mention().is_some() => self.at_complete(),
            KeyCode::Char(c) if !ctrl && !alt => {
                self.editor.insert(c);
                if c == '/' || self.slash_active() {
                    self.slash_sel = 0;
                }
                if c == '@' {
                    self.spawn_at_listing();
                }
                if self.editor.mention().is_some() {
                    self.at_sel = 0;
                }
            }
            _ => {}
        }
    }

    /// Suspend the TUI, edit the current input in `$VISUAL`/`$EDITOR`
    /// (falling back to `vi`), and load the result back on a clean exit.
    fn open_external_editor(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
        kitty: bool,
    ) {
        let path = std::env::temp_dir().join(format!("m-input-{}.md", std::process::id()));
        if std::fs::write(&path, self.editor.text()).is_err() {
            return;
        }
        let editor_cmd = std::env::var("VISUAL")
            .or_else(|_| std::env::var("EDITOR"))
            .unwrap_or_else(|_| "vi".to_string());
        let mut parts = editor_cmd.split_whitespace();
        let Some(prog) = parts.next() else {
            std::fs::remove_file(&path).ok();
            return;
        };
        let args: Vec<&str> = parts.collect();

        restore_terminal(kitty);
        let result = std::process::Command::new(prog)
            .args(&args)
            .arg(&path)
            .status();
        enter_terminal(kitty).ok();
        terminal.clear().ok();

        match result {
            Ok(status) if status.success() => {
                if let Ok(new_text) = std::fs::read_to_string(&path) {
                    self.editor.set(new_text.trim_end_matches('\n'));
                }
            }
            Ok(_) => {} // editor exited non-zero (e.g. `:cq`): keep the original input
            Err(e) => self.notice(format!("failed to launch editor '{prog}': {e}")),
        }
        std::fs::remove_file(&path).ok();
        self.dirty = true;
    }

    /// Hot-reload: re-exec the freshly built binary in place, resuming this
    /// exact session. On success this never returns (the process image is
    /// replaced).
    ///
    /// Deliberately *not* `std::env::current_exe()`: `cargo build --release`
    /// replaces `target/release/m` by rename, and on Linux `/proc/self/exe`
    /// keeps pointing at the old, now-unlinked inode — `current_exe()`
    /// reports it as `<path> (deleted)` and exec fails. `release_binary_path`
    /// is a plain path computed from the workspace this copy was built
    /// from, so the exec looks up whatever is *currently* linked there.
    ///
    /// Restores the tty *before* exec. The replacement process calls
    /// `enable_raw_mode` and treats whatever termios it sees as the
    /// "original" cooked mode — if we exec while still raw, quit later
    /// restores to raw and the parent shell loses echo.
    fn reload(&mut self) {
        let exe = release_binary_path()
            .filter(|p| p.exists())
            .or_else(|| std::env::current_exe().ok());
        let Some(exe) = exe else {
            self.notice("reload failed: could not locate the m binary");
            self.dirty = true;
            return;
        };
        let kitty = self.kitty;
        restore_terminal(kitty);
        use std::os::unix::process::CommandExt;
        let err = std::process::Command::new(&exe)
            .arg("-C")
            .arg(&self.cwd)
            .arg("-m")
            .arg(&self.profile_name)
            .arg("--session")
            .arg(&self.session_path)
            .exec();
        // exec failed — get the TUI back so the user can keep working.
        if let Err(e) = enter_terminal(kitty) {
            self.notice(format!(
                "reload failed to exec {} ({err}); also re-enter TUI: {e}",
                exe.display()
            ));
        } else {
            self.notice(format!("reload failed to exec {}: {err}", exe.display()));
        }
        self.dirty = true;
    }

    /// `cargo build --release` then `cargo test --release` in `self.cwd`,
    /// off the UI thread — a build that compiles but fails its own tests
    /// never gets hot-loaded over a working session. Both commands' own
    /// stdout/stderr are captured (not inherited) so they don't corrupt the
    /// alternate screen. `apply()` schedules a `reload()` once
    /// `UiMsg::RebuildDone(Ok(()))` comes back.
    fn spawn_rebuild(&mut self) {
        self.notice("rebuilding in the background (cargo build --release)…");
        let cwd = self.cwd.clone();
        let tx = self.ui_tx.clone();
        std::thread::spawn(move || {
            let msg = run_cargo_step(&cwd, &["build", "--release"], "build")
                .and_then(|()| run_cargo_step(&cwd, &["test", "--release"], "tests"));
            tx.send(UiMsg::RebuildDone(msg)).ok();
        });
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

    /// Refresh the project file listing in the background; called on every
    /// literal `@` keystroke rather than per filter-keystroke, since
    /// filtering itself is a pure in-memory pass over `at_files`.
    fn spawn_at_listing(&mut self) {
        self.at_loading = true;
        let cwd = self.cwd.clone();
        let tx = self.ui_tx.clone();
        std::thread::spawn(move || {
            tx.send(UiMsg::AtFiles(files::list_project_files(&cwd)))
                .ok();
        });
    }

    /// Whether the `@`-mention popup should be showing right now.
    fn at_open(&self) -> bool {
        matches!(self.editor.mention(), Some((start, _)) if self.at_dismissed_at != Some(start))
    }

    fn at_matches(&self) -> Vec<(String, i64)> {
        let Some((_, query)) = self.editor.mention() else {
            return Vec::new();
        };
        let mut out: Vec<(String, i64)> = self
            .at_files
            .iter()
            .filter_map(|f| fuzzy::score(f, query).map(|s| (f.clone(), s)))
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.len().cmp(&b.0.len())));
        out.truncate(20);
        out
    }

    /// Insert the selected match as `@relative/path ` (trailing space).
    /// That trailing space naturally closes the popup on the next frame —
    /// `Editor::mention` sees whitespace in the query and returns `None` —
    /// so there's no separate "close" step needed.
    fn at_complete(&mut self) {
        let Some((start, _)) = self.editor.mention() else {
            return;
        };
        let matches = self.at_matches();
        if let Some((path, _)) = matches.get(self.at_sel.min(matches.len().saturating_sub(1))) {
            self.editor.complete_mention(start, &format!("@{path} "));
        }
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
                     ctrl+x ctrl+e edit in $EDITOR · @ file picker · \
                     pgup/pgdn or wheel scroll · \
                     /new /resume /compact /model <provider>/<model> /reload /rebuild /quit",
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
            "/model" => {
                if args.is_empty() {
                    let current = self.profile_label.clone();
                    let list = m_core::config::list_selections()
                        .into_iter()
                        .map(|(p, m)| {
                            let label = format!("{p}/{m}");
                            if label == current {
                                format!("{label} (current)")
                            } else {
                                label
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.notice(format!(
                        "models: {list} — /model <provider>/<model> or /model <provider> <model>"
                    ));
                } else if self.running {
                    self.notice("busy — esc to cancel first");
                } else {
                    self.cmd_tx
                        .send(AgentCmd::SwitchProfile(args.to_string()))
                        .ok();
                }
            }
            "/reload" => {
                if self.running {
                    self.notice("busy — esc to cancel first");
                } else {
                    self.reload();
                }
            }
            "/rebuild" => {
                if self.running {
                    self.notice("busy — esc to cancel first");
                } else {
                    self.spawn_rebuild();
                }
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
        if self.at_open() {
            self.draw_at_menu(f, input_area);
        } else if self.slash_active() {
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
        let mut block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(border_style);
        if self.leader_x.is_some() {
            block = block.title(Span::styled(
                " ctrl+x ctrl+e: edit in $EDITOR ",
                theme::dim(),
            ));
        }
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

    fn draw_at_menu(&mut self, f: &mut Frame, input_area: Rect) {
        let matches = self.at_matches();
        self.at_sel = self.at_sel.min(matches.len().saturating_sub(1));
        let loading = self.at_loading && self.at_files.is_empty();
        let h = matches.len().max(1) as u16 + 2;
        let area = Rect {
            x: input_area.x + 2,
            y: input_area.y.saturating_sub(h),
            width: 60.min(f.area().width.saturating_sub(4)),
            height: h,
        };
        let lines: Vec<Line> = if matches.is_empty() {
            let msg = if loading {
                "loading files…"
            } else {
                "no matches"
            };
            vec![Line::styled(format!(" {msg}"), theme::dim())]
        } else {
            matches
                .iter()
                .enumerate()
                .map(|(i, (path, _))| {
                    let style = if i == self.at_sel {
                        theme::accent().reversed()
                    } else {
                        Style::default()
                    };
                    Line::styled(format!(" {path}"), style)
                })
                .collect()
        };
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

/// Outcome of feeding a keypress through the emacs-style `ctrl+x` leader
/// prefix used to open the input in `$EDITOR` (`ctrl+x ctrl+e`, mirroring
/// bash readline's edit-and-execute-command).
#[derive(Debug, PartialEq, Eq)]
enum LeaderOutcome {
    /// No leader sequence involved; caller handles the key as usual.
    Pass,
    /// `ctrl+x` seen; now waiting for the completing key.
    Armed,
    /// `ctrl+x` followed by `e`/`E` within the timeout: open the editor.
    OpenEditor,
}

fn leader_step(pending: bool, ctrl: bool, code: KeyCode) -> LeaderOutcome {
    if pending {
        return match code {
            KeyCode::Char('e') | KeyCode::Char('E') => LeaderOutcome::OpenEditor,
            _ => LeaderOutcome::Pass,
        };
    }
    if ctrl && matches!(code, KeyCode::Char('x')) {
        return LeaderOutcome::Armed;
    }
    LeaderOutcome::Pass
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

#[cfg(test)]
mod leader_tests {
    use super::*;

    #[test]
    fn ctrl_x_arms_the_leader() {
        assert_eq!(
            leader_step(false, true, KeyCode::Char('x')),
            LeaderOutcome::Armed
        );
    }

    #[test]
    fn plain_x_does_not_arm() {
        assert_eq!(
            leader_step(false, false, KeyCode::Char('x')),
            LeaderOutcome::Pass
        );
    }

    #[test]
    fn e_after_leader_opens_editor_ctrl_or_not() {
        assert_eq!(
            leader_step(true, false, KeyCode::Char('e')),
            LeaderOutcome::OpenEditor
        );
        assert_eq!(
            leader_step(true, true, KeyCode::Char('e')),
            LeaderOutcome::OpenEditor
        );
        assert_eq!(
            leader_step(true, false, KeyCode::Char('E')),
            LeaderOutcome::OpenEditor
        );
    }

    #[test]
    fn other_key_after_leader_passes_through_instead_of_opening() {
        assert_eq!(
            leader_step(true, false, KeyCode::Char('q')),
            LeaderOutcome::Pass
        );
        assert_eq!(
            leader_step(true, true, KeyCode::Char('c')),
            LeaderOutcome::Pass
        );
    }

    #[test]
    fn second_ctrl_x_while_not_pending_rearms() {
        assert_eq!(
            leader_step(false, true, KeyCode::Char('x')),
            LeaderOutcome::Armed
        );
    }
}
