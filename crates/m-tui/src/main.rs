//! m — minimal, fast coding agent. Interactive TUI by default;
//! `-p` for headless print mode (scriptable, used by the bench runner).

mod tui;

use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use m_core::agent::{Agent, AgentEvent, StopReason};
use m_core::context::SkillInfo;
use m_core::{config, context, prompt, session::Session};

/// System prompt + discovered skills, shared by print mode and the TUI.
/// Loads hierarchical AGENTS.md/CLAUDE.md and scans skill directories
/// (including ~/.claude/skills for Claude Code interop).
pub fn build_env(cwd: &Path) -> (String, Vec<SkillInfo>) {
    let project_context = context::load_project_context(cwd);
    let skills = context::discover_skills(cwd);
    let index = context::skills_index(&skills);
    (prompt::system_prompt(cwd, &project_context, &index), skills)
}

const USAGE: &str = "\
m — minimal, fast coding agent

Usage:
  m [options]                interactive session (TUI)
  m -p [prompt] [options]    print mode: run one prompt, stream to stdout
                             (prompt read from stdin if omitted)

Options:
  -p, --print [PROMPT]   headless mode
      --json             with -p: emit events as JSON lines
  -m, --profile NAME     provider profile from ~/.config/m/config.toml
  -r, --resume           continue the most recent session in this directory
      --session PATH     resume a specific session file
  -C, --dir PATH         working directory for the agent
      --max-turns N      stop after N model turns (0 = unlimited)
      --max-tokens N     cap tokens per model response
      --temp F           sampling temperature (default: server/profile default)
  -V, --version          print version
  -h, --help             this help
";

struct Args {
    print: bool,
    json: bool,
    prompt: Option<String>,
    profile: Option<String>,
    resume: bool,
    session: Option<PathBuf>,
    dir: Option<PathBuf>,
    max_turns: usize,
    max_tokens: Option<u32>,
    temp: Option<f32>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        print: false,
        json: false,
        prompt: None,
        profile: None,
        resume: false,
        session: None,
        dir: None,
        max_turns: 0,
        max_tokens: None,
        temp: None,
    };
    let mut it = std::env::args().skip(1).peekable();
    let mut positionals: Vec<String> = Vec::new();
    while let Some(arg) = it.next() {
        let mut need = |name: &str| it.next().ok_or(format!("{name} requires a value"));
        match arg.as_str() {
            "-p" | "--print" => a.print = true,
            "--json" => a.json = true,
            "-m" | "--profile" => a.profile = Some(need("--profile")?),
            "-r" | "--resume" => a.resume = true,
            "--session" => a.session = Some(PathBuf::from(need("--session")?)),
            "-C" | "--dir" => a.dir = Some(PathBuf::from(need("--dir")?)),
            "--max-turns" => {
                a.max_turns = need("--max-turns")?
                    .parse()
                    .map_err(|_| "bad --max-turns")?
            }
            "--max-tokens" => {
                a.max_tokens = Some(
                    need("--max-tokens")?
                        .parse()
                        .map_err(|_| "bad --max-tokens")?,
                )
            }
            "--temp" => a.temp = Some(need("--temp")?.parse().map_err(|_| "bad --temp")?),
            "-V" | "--version" => {
                println!("m {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            s if s.starts_with('-') => return Err(format!("unknown flag: {s} (see m --help)")),
            s => positionals.push(s.to_string()),
        }
    }
    if !positionals.is_empty() {
        a.prompt = Some(positionals.join(" "));
    }
    Ok(a)
}

fn main() {
    let t0 = std::time::Instant::now();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("m: {e}");
            std::process::exit(2);
        }
    };

    let cwd = match &args.dir {
        Some(d) => match d.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                eprintln!("m: --dir {}: {e}", d.display());
                std::process::exit(2);
            }
        },
        None => std::env::current_dir().expect("cwd"),
    };

    let mut cfg = match config::load(args.profile.as_deref()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("m: {e}");
            std::process::exit(2);
        }
    };
    cfg.max_turns = args.max_turns;
    if args.temp.is_some() {
        cfg.profile.temperature = args.temp;
    }
    if args.max_tokens.is_some() {
        cfg.profile.max_tokens = args.max_tokens;
    }

    if args.print {
        std::process::exit(run_print(args, cfg, cwd));
    }

    let resume = if args.resume || args.session.is_some() {
        let p = args.session.clone().or_else(|| Session::latest(&cwd));
        if p.is_none() {
            eprintln!("m: no session to resume in this directory");
            std::process::exit(2);
        }
        p
    } else {
        None
    };
    match tui::run_tui(cfg, cwd, resume, t0) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("m: {e}");
            std::process::exit(1);
        }
    }
}

fn run_print(args: Args, cfg: config::Config, cwd: PathBuf) -> i32 {
    let prompt_text = match args.prompt {
        Some(p) => p,
        None => {
            let mut s = String::new();
            if std::io::stdin().read_to_string(&mut s).is_err() || s.trim().is_empty() {
                eprintln!("m: no prompt given (pass as argument or on stdin)");
                return 2;
            }
            s.trim().to_string()
        }
    };

    let (sys, skills) = build_env(&cwd);
    let agent = if args.resume || args.session.is_some() {
        let path = args.session.clone().or_else(|| Session::latest(&cwd));
        match path {
            Some(p) => Agent::resume(cfg, cwd, sys, skills, &p),
            None => {
                eprintln!("m: no session to resume in this directory");
                return 2;
            }
        }
    } else {
        Agent::new(cfg, cwd, sys, skills)
    };
    let mut agent = match agent {
        Ok(a) => a,
        Err(e) => {
            eprintln!("m: {e}");
            return 2;
        }
    };

    // Ctrl+C sets the cancel flag; second Ctrl+C kills the process.
    install_sigint(agent.cancel.clone());

    let json = args.json;
    let color = std::io::stderr().is_terminal();
    let mut on_event = move |ev: AgentEvent| {
        if json {
            print_json_event(&ev);
            return;
        }
        match ev {
            AgentEvent::Content(s) => {
                print!("{s}");
                std::io::stdout().flush().ok();
            }
            AgentEvent::Reasoning(_) => {}
            AgentEvent::ToolStart { name, args } => {
                let summary = summarize_args(&name, &args);
                if color {
                    eprintln!("\x1b[2m▸ {name} {summary}\x1b[0m");
                } else {
                    eprintln!("* {name} {summary}");
                }
            }
            AgentEvent::ToolEnd {
                is_error, output, ..
            } => {
                if is_error {
                    let first = output.lines().next().unwrap_or("");
                    if color {
                        eprintln!("\x1b[2m  ! {}\x1b[0m", m_core::http::truncate(first, 160));
                    } else {
                        eprintln!("  ! {}", m_core::http::truncate(first, 160));
                    }
                }
            }
            AgentEvent::UserInjected(_) => {}
            AgentEvent::Telemetry { .. } => {}
            AgentEvent::Notice(n) => eprintln!("m: {n}"),
        }
    };

    match agent.run_prompt(&prompt_text, &mut on_event) {
        Ok(outcome) => {
            if !json {
                // Ensure the final answer ends with a newline on stdout.
                println!();
            }
            match outcome.stop {
                StopReason::Done => 0,
                StopReason::Length => {
                    eprintln!("m: stopped: token limit");
                    1
                }
                StopReason::MaxTurns => {
                    eprintln!("m: stopped: reached --max-turns");
                    1
                }
                StopReason::Cancelled => {
                    eprintln!("m: cancelled");
                    130
                }
            }
        }
        Err(e) => {
            eprintln!("m: {e}");
            1
        }
    }
}

pub fn summarize_args(name: &str, args: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(args).unwrap_or_default();
    let s = match name {
        "bash" => v
            .get("command")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string(),
        "skill" => v
            .get("name")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string(),
        _ => v
            .get("path")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string(),
    };
    m_core::http::truncate(s.replace('\n', " ⏎ ").trim(), 120)
}

fn print_json_event(ev: &AgentEvent) {
    use serde_json::json;
    let v = match ev {
        AgentEvent::Reasoning(s) => json!({"type": "reasoning", "text": s}),
        AgentEvent::Content(s) => json!({"type": "content", "text": s}),
        AgentEvent::ToolStart { name, args } => {
            json!({"type": "tool_start", "name": name, "args": args})
        }
        AgentEvent::ToolEnd {
            name,
            output,
            is_error,
            ..
        } => {
            json!({"type": "tool_end", "name": name, "output": output, "is_error": is_error})
        }
        AgentEvent::UserInjected(s) => json!({"type": "user_injected", "text": s}),
        AgentEvent::Telemetry { usage, timings } => json!({
            "type": "telemetry",
            "prompt_tokens": usage.as_ref().map(|u| u.prompt_tokens),
            "completion_tokens": usage.as_ref().map(|u| u.completion_tokens),
            "tok_per_sec": timings.as_ref().map(|t| t.predicted_per_second),
            "draft_accept": timings.as_ref().map(|t| {
                if t.draft_n > 0 { t.draft_n_accepted as f64 / t.draft_n as f64 } else { 0.0 }
            }),
            "cached_tokens": timings.as_ref().map(|t| t.cache_n),
        }),
        AgentEvent::Notice(s) => json!({"type": "notice", "text": s}),
    };
    println!("{v}");
}

fn install_sigint(cancel: std::sync::Arc<std::sync::atomic::AtomicBool>) {
    use std::sync::OnceLock;
    use std::sync::atomic::AtomicBool;
    static CANCEL: OnceLock<std::sync::Arc<AtomicBool>> = OnceLock::new();
    static ARMED: AtomicBool = AtomicBool::new(false);
    let _ = CANCEL.set(cancel);
    extern "C" fn handler(_: libc::c_int) {
        // Second Ctrl+C: hard exit (async-signal-safe _exit).
        if ARMED.swap(true, Ordering::SeqCst) {
            unsafe { libc::_exit(130) }
        }
        if let Some(c) = CANCEL.get() {
            c.store(true, Ordering::SeqCst);
        }
    }
    unsafe {
        let f: extern "C" fn(libc::c_int) = handler;
        libc::signal(libc::SIGINT, f as usize);
    }
}
