//! The agent loop: send messages → stream response → run tool calls →
//! repeat until the model answers without tools. pi's loop, nothing more.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::config::Config;
use crate::context::SkillInfo;
use crate::error::{Error, Result};
use crate::provider::{self, Delta, Msg, Timings, ToolSpec, Usage};
use crate::session::Session;
use crate::tools;

/// Events surfaced to the front end (print mode or TUI).
#[derive(Debug)]
pub enum AgentEvent {
    Reasoning(String),
    Content(String),
    /// A tool call is about to execute (arguments fully received).
    ToolStart { name: String, args: String },
    /// `detail` is UI-only extra (a diff for edit/write); not model-visible.
    ToolEnd { name: String, output: String, is_error: bool, detail: Option<String> },
    /// Queued steering input was injected into the conversation.
    UserInjected(String),
    Telemetry { usage: Option<Usage>, timings: Option<Timings> },
    /// Non-fatal problem worth showing (retry, truncation, …).
    Notice(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StopReason {
    /// Model produced a final answer.
    Done,
    Cancelled,
    MaxTurns,
    /// Response cut off by token limit.
    Length,
}

#[derive(Debug)]
pub struct RunOutcome {
    pub stop: StopReason,
    pub final_text: String,
    pub turns: usize,
}

/// Fraction of the context window at which old tool outputs get clipped
/// (in memory only — the session file stays faithful).
const CTX_GUARD_FRACTION: f64 = 0.85;
/// Never clip messages within this many of the end.
const CTX_GUARD_KEEP_TAIL: usize = 8;

pub struct Agent {
    pub config: Config,
    pub session: Session,
    pub cwd: PathBuf,
    pub cancel: Arc<AtomicBool>,
    /// User input queued mid-run; injected before the next model call.
    pub steer: Arc<Mutex<VecDeque<String>>>,
    system_prompt: String,
    skills: Vec<SkillInfo>,
    tools: Vec<ToolSpec>,
    /// Context size, possibly updated by a background /props probe.
    ctx_limit: Arc<AtomicUsize>,
    last_prompt_tokens: u64,
}

impl Agent {
    pub fn new(
        config: Config,
        cwd: PathBuf,
        system_prompt: String,
        skills: Vec<SkillInfo>,
    ) -> Result<Agent> {
        let session = Session::new(&cwd, &config.profile.model)?;
        Ok(Agent::with_session(config, cwd, system_prompt, skills, session))
    }

    pub fn resume(
        config: Config,
        cwd: PathBuf,
        system_prompt: String,
        skills: Vec<SkillInfo>,
        path: &Path,
    ) -> Result<Agent> {
        let session = Session::load(path)?;
        Ok(Agent::with_session(config, cwd, system_prompt, skills, session))
    }

    fn with_session(
        config: Config,
        cwd: PathBuf,
        system_prompt: String,
        skills: Vec<SkillInfo>,
        session: Session,
    ) -> Agent {
        let ctx_limit = Arc::new(AtomicUsize::new(config.profile.ctx));
        // The local server knows its real context size; probe off the hot path.
        {
            let ctx_limit = Arc::clone(&ctx_limit);
            let profile = config.profile.clone();
            std::thread::spawn(move || {
                if let Some(n) = provider::probe_ctx(&profile) {
                    ctx_limit.store(n, Ordering::Relaxed);
                }
            });
        }
        Agent {
            config,
            session,
            cwd,
            cancel: Arc::new(AtomicBool::new(false)),
            steer: Arc::new(Mutex::new(VecDeque::new())),
            system_prompt,
            tools: tools::specs(!skills.is_empty()),
            skills,
            ctx_limit,
            last_prompt_tokens: 0,
        }
    }

    /// Context-limit handle (updated by the background /props probe).
    pub fn ctx_limit_handle(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.ctx_limit)
    }

    /// Start a fresh session (new file, empty history).
    pub fn new_session(&mut self) -> Result<()> {
        self.session = Session::new(&self.cwd, &self.config.profile.model)?;
        self.last_prompt_tokens = 0;
        Ok(())
    }

    /// Switch to an existing session file.
    pub fn load_session(&mut self, path: &Path) -> Result<()> {
        self.session = Session::load(path)?;
        self.last_prompt_tokens = 0;
        Ok(())
    }

    /// Summarize the session with the model, then continue in a fresh
    /// session seeded with the summary (frees the context window).
    pub fn compact(&mut self, on_event: &mut dyn FnMut(AgentEvent)) -> Result<()> {
        if self.session.messages.is_empty() {
            return Err(Error::msg("nothing to compact"));
        }
        let mut wire = Vec::with_capacity(self.session.messages.len() + 2);
        wire.push(Msg::system(&self.system_prompt));
        wire.extend(self.session.messages.iter().cloned());
        wire.push(Msg::user(
            "Summarize this session so work can continue in a fresh context. Include: the task \
             and its current state, key decisions, relevant file paths and their roles, and any \
             unfinished steps. Be concise but complete. Reply with only the summary.",
        ));
        let completion = provider::stream_chat(
            &self.config.profile,
            &wire,
            &[],
            self.config.profile.temperature,
            Arc::clone(&self.cancel),
            |d| {
                if let Delta::Content(s) = d {
                    on_event(AgentEvent::Content(s));
                }
            },
        )?;
        let summary = completion.msg.content.unwrap_or_default();
        if summary.trim().is_empty() {
            return Err(Error::msg("empty summary; session left unchanged"));
        }
        self.new_session()?;
        self.session.push(Msg::user(format!(
            "[Continuing from a compacted session. Summary of prior context:]\n\n{summary}"
        )))?;
        self.session.push(Msg {
            role: "assistant".into(),
            content: Some("Understood — continuing from that state.".into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        })?;
        Ok(())
    }

    /// Run one user prompt to completion (or cancellation / turn cap).
    pub fn run_prompt(
        &mut self,
        prompt: &str,
        on_event: &mut dyn FnMut(AgentEvent),
    ) -> Result<RunOutcome> {
        self.session.push(Msg::user(prompt))?;
        self.run_loop(on_event)
    }

    fn run_loop(&mut self, on_event: &mut dyn FnMut(AgentEvent)) -> Result<RunOutcome> {
        const MAX_LENGTH_NUDGES: usize = 3;
        let mut turns = 0usize;
        let mut nudges = 0usize;
        loop {
            if self.cancel.load(Ordering::Relaxed) {
                return Ok(RunOutcome { stop: StopReason::Cancelled, final_text: String::new(), turns });
            }
            self.guard_context(on_event);

            let mut wire = Vec::with_capacity(self.session.messages.len() + 1);
            wire.push(Msg::system(&self.system_prompt));
            wire.extend(self.session.messages.iter().cloned());

            let completion = match self.stream_with_retry(&wire, on_event) {
                Ok(c) => c,
                Err(Error::Cancelled) => {
                    return Ok(RunOutcome {
                        stop: StopReason::Cancelled,
                        final_text: String::new(),
                        turns,
                    });
                }
                Err(e) => return Err(e),
            };

            if let Some(u) = &completion.usage {
                self.last_prompt_tokens = u.prompt_tokens + u.completion_tokens;
            }
            on_event(AgentEvent::Telemetry {
                usage: completion.usage.clone(),
                timings: completion.timings.clone(),
            });

            self.session.push(completion.msg.clone())?;
            turns += 1;

            let tool_calls = completion.msg.tool_calls.clone().unwrap_or_default();
            if tool_calls.is_empty() {
                if completion.finish_reason == "length" {
                    // Truncated mid-thought (usually a reasoning runaway on
                    // small models). Nudge it back on track a few times
                    // before giving up.
                    if nudges < MAX_LENGTH_NUDGES {
                        nudges += 1;
                        on_event(AgentEvent::Notice(format!(
                            "response hit the token limit — nudging ({nudges}/{MAX_LENGTH_NUDGES})"
                        )));
                        self.session.push(Msg::user(
                            "(Your response was cut off at the token limit. Stop deliberating: \
                             take the next concrete step with a single tool call, or give your \
                             final answer in a few sentences.)",
                        ))?;
                        continue;
                    }
                    on_event(AgentEvent::Notice("response hit the token limit".into()));
                    return Ok(RunOutcome {
                        stop: StopReason::Length,
                        final_text: completion.msg.content.unwrap_or_default(),
                        turns,
                    });
                }
                return Ok(RunOutcome {
                    stop: StopReason::Done,
                    final_text: completion.msg.content.unwrap_or_default(),
                    turns,
                });
            }

            for call in &tool_calls {
                if self.cancel.load(Ordering::Relaxed) {
                    // Every issued call needs a tool result or the next
                    // request is malformed.
                    self.session.push(Msg::tool_result(&call.id, "Cancelled by user."))?;
                    continue;
                }
                on_event(AgentEvent::ToolStart {
                    name: call.function.name.clone(),
                    args: call.function.arguments.clone(),
                });
                let out = tools::execute(
                    &call.function.name,
                    &call.function.arguments,
                    &self.cwd,
                    &self.skills,
                    &self.cancel,
                );
                on_event(AgentEvent::ToolEnd {
                    name: call.function.name.clone(),
                    output: out.content.clone(),
                    is_error: out.is_error,
                    detail: out.detail,
                });
                self.session.push(Msg::tool_result(&call.id, out.content))?;
            }

            // Inject any queued steering input.
            let queued: Vec<String> = self.steer.lock().unwrap().drain(..).collect();
            for text in queued {
                on_event(AgentEvent::UserInjected(text.clone()));
                self.session.push(Msg::user(text))?;
            }

            if self.config.max_turns > 0 && turns >= self.config.max_turns {
                return Ok(RunOutcome {
                    stop: StopReason::MaxTurns,
                    final_text: String::new(),
                    turns,
                });
            }
        }
    }

    /// One retry on transport-level failures (the local server restarts,
    /// hosted endpoints hiccup); API 4xx errors are not retried.
    fn stream_with_retry(
        &self,
        wire: &[Msg],
        on_event: &mut dyn FnMut(AgentEvent),
    ) -> Result<provider::Completion> {
        fn forward(d: Delta, on_event: &mut dyn FnMut(AgentEvent)) {
            match d {
                Delta::Reasoning(s) => on_event(AgentEvent::Reasoning(s)),
                Delta::Content(s) => on_event(AgentEvent::Content(s)),
                Delta::ToolCallBegin { .. } => {}
            }
        }
        let mut attempt = 0;
        loop {
            let result = provider::stream_chat(
                &self.config.profile,
                wire,
                &self.tools,
                self.config.profile.temperature,
                Arc::clone(&self.cancel),
                |d| forward(d, on_event),
            );
            match result {
                Err(Error::Msg(m)) if attempt == 0 && !m.starts_with("API error") => {
                    on_event(AgentEvent::Notice(format!("{m} — retrying")));
                    attempt += 1;
                    std::thread::sleep(Duration::from_secs(1));
                }
                other => return other,
            }
        }
    }

    /// When the conversation nears the context limit, clip old tool outputs
    /// in memory. This costs the KV prefix cache once, but keeps long runs
    /// alive.
    fn guard_context(&mut self, on_event: &mut dyn FnMut(AgentEvent)) {
        let limit = self.ctx_limit.load(Ordering::Relaxed);
        if limit == 0 || (self.last_prompt_tokens as f64) < limit as f64 * CTX_GUARD_FRACTION {
            return;
        }
        let n = self.session.messages.len();
        let mut clipped = 0usize;
        for msg in self.session.messages.iter_mut().take(n.saturating_sub(CTX_GUARD_KEEP_TAIL)) {
            if msg.role == "tool"
                && let Some(c) = &msg.content
                && c.len() > 600
            {
                let head: String = c.chars().take(200).collect();
                *msg = Msg {
                    content: Some(format!("{head}\n(… clipped to fit context …)")),
                    ..msg.clone()
                };
                clipped += 1;
            }
        }
        if clipped > 0 {
            on_event(AgentEvent::Notice(format!(
                "context {}% full — clipped {clipped} old tool outputs",
                (self.last_prompt_tokens as f64 / limit as f64 * 100.0) as u32
            )));
            // Force re-measure on the next response.
            self.last_prompt_tokens = 0;
        }
    }
}
