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
use crate::provider::{self, ChatProvider, Delta, Msg, Timings, ToolSpec, Usage};
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
/// Minimum sampling temperature for the one request after a recovery event
/// (length nudge or annotated identical repeat). At temp 0 a loop is a
/// fixed point — the same context reproduces the same runaway; resampling
/// is the way out. Applied only when the configured temperature is lower.
const RECOVERY_TEMP: f32 = 0.4;

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
    /// The chat endpoint. HTTP in production; tests inject scripted
    /// completions here to exercise the loop without a server.
    provider: Box<dyn ChatProvider>,
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
            provider: Box::new(provider::Http),
            ctx_limit,
            last_prompt_tokens: 0,
        }
    }

    /// Replace the chat provider (tests, alternative transports).
    pub fn set_provider(&mut self, provider: Box<dyn ChatProvider>) {
        self.provider = provider;
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
        let completion = self.provider.stream_chat(
            &self.config.profile,
            &wire,
            &[],
            self.config.profile.temperature,
            Arc::clone(&self.cancel),
            &mut |d| {
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
        const MAX_LENGTH_NUDGES: usize = 5;
        let mut turns = 0usize;
        let mut nudges = 0usize;
        // Repeated identical tool calls: signature → (executions, last output
        // hash). Cleared whenever a write/edit succeeds (state changed, so a
        // rerun is legitimate again).
        let mut seen: std::collections::HashMap<u64, (u32, u64)> = Default::default();
        // Set when the previous turn hit a recovery event; bumps the next
        // request's temperature once (see RECOVERY_TEMP).
        let mut recovery = false;
        loop {
            if self.cancel.load(Ordering::Relaxed) {
                return Ok(RunOutcome { stop: StopReason::Cancelled, final_text: String::new(), turns });
            }
            self.guard_context(on_event);

            let mut wire = Vec::with_capacity(self.session.messages.len() + 1);
            wire.push(Msg::system(&self.system_prompt));
            wire.extend(self.session.messages.iter().cloned());

            let temperature = if std::mem::take(&mut recovery) {
                let t = Self::recovery_temp(self.config.profile.temperature);
                if t != self.config.profile.temperature {
                    on_event(AgentEvent::Notice(format!(
                        "recovery turn — temperature {RECOVERY_TEMP} for one request"
                    )));
                }
                t
            } else {
                self.config.profile.temperature
            };

            let completion = match self.stream_with_retry(&wire, temperature, on_event) {
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

            turns += 1;

            let tool_calls = completion.msg.tool_calls.clone().unwrap_or_default();
            if tool_calls.is_empty() {
                if completion.finish_reason == "length" {
                    // Truncated mid-thought (usually a reasoning runaway on
                    // small models). Retry fresh: the truncated message is
                    // NOT added to the context — resending the runaway text
                    // reliably re-triggers the same loop — only the nudge is.
                    if nudges < MAX_LENGTH_NUDGES {
                        nudges += 1;
                        recovery = true;
                        on_event(AgentEvent::Notice(format!(
                            "response hit the token limit — retrying ({nudges}/{MAX_LENGTH_NUDGES})"
                        )));
                        self.session.push(Msg::user(
                            "(Your previous response overran the token limit and was discarded. \
                             Do not repeat that reasoning. Take the next concrete step with a \
                             single tool call, or give your final answer in a few sentences.)",
                        ))?;
                        continue;
                    }
                    self.session.push(completion.msg.clone())?;
                    on_event(AgentEvent::Notice("response hit the token limit".into()));
                    return Ok(RunOutcome {
                        stop: StopReason::Length,
                        final_text: completion.msg.content.unwrap_or_default(),
                        turns,
                    });
                }
                self.session.push(completion.msg.clone())?;
                return Ok(RunOutcome {
                    stop: StopReason::Done,
                    final_text: completion.msg.content.unwrap_or_default(),
                    turns,
                });
            }
            self.session.push(completion.msg.clone())?;

            for call in &tool_calls {
                if self.cancel.load(Ordering::Relaxed) {
                    // Every issued call needs a tool result or the next
                    // request is malformed.
                    self.session.push(Msg::tool_result(&call.id, "Cancelled by user."))?;
                    continue;
                }
                // Repeat detection. Deliberately NOT a blocker: an A/B on the
                // held-out bench slice showed that refusing to execute makes
                // temp-0 loops stickier (the model loops on the refusal, and
                // a frozen context is a fixed point). Executing keeps the
                // context evolving; the annotation gives the model a way out.
                let sig = Self::fnv(&call.function.name, &call.function.arguments);
                on_event(AgentEvent::ToolStart {
                    name: call.function.name.clone(),
                    args: call.function.arguments.clone(),
                });
                let mut out = tools::execute(
                    &call.function.name,
                    &call.function.arguments,
                    &self.cwd,
                    &self.skills,
                    &self.cancel,
                );
                let out_hash = Self::fnv("", &out.content);
                match seen.entry(sig) {
                    std::collections::hash_map::Entry::Occupied(mut e) => {
                        let (n, prev) = e.get_mut();
                        *n += 1;
                        if *prev == out_hash {
                            recovery = true;
                            out.content.push_str(&format!(
                                "\n(note: you have run exactly this {n} times now with \
                                 identical output — question the hypothesis that led here \
                                 and try a different approach)",
                            ));
                        }
                        *prev = out_hash;
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert((1, out_hash));
                    }
                }
                // A successful mutation makes reruns legitimate again.
                if !out.is_error && matches!(call.function.name.as_str(), "write" | "edit") {
                    seen.clear();
                }
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
        temperature: Option<f32>,
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
            let result = self.provider.stream_chat(
                &self.config.profile,
                wire,
                &self.tools,
                temperature,
                Arc::clone(&self.cancel),
                &mut |d| forward(d, on_event),
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

    /// The temperature for a recovery request: at least RECOVERY_TEMP when
    /// an explicit lower temperature is configured; a server-default (None)
    /// or already-hot temperature is left alone.
    fn recovery_temp(configured: Option<f32>) -> Option<f32> {
        match configured {
            Some(t) if t < RECOVERY_TEMP => Some(RECOVERY_TEMP),
            other => other,
        }
    }

    fn fnv(a: &str, b: &str) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for byte in a.bytes().chain([0u8]).chain(b.bytes()) {
            h ^= byte as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{Completion, FunctionCall, ToolCall};
    use std::sync::atomic::AtomicUsize;

    /// Scripted provider: pops one pre-baked completion per request and
    /// records every wire it was sent, so tests can assert on exactly what
    /// the model would have seen.
    struct Scripted {
        replies: Mutex<VecDeque<Result<Completion>>>,
        wires: Arc<Mutex<Vec<Vec<Msg>>>>,
        temps: Arc<Mutex<Vec<Option<f32>>>>,
        /// Set the cancel flag while serving this (1-based) request.
        cancel_on_call: Option<usize>,
    }

    /// scriptable_agent return type: (agent, wire history, temperature history).
    type ScriptedTuple = (Agent, Arc<Mutex<Vec<Vec<Msg>>>>, Arc<Mutex<Vec<Option<f32>>>>);

    impl ChatProvider for Scripted {
        fn stream_chat(
            &self,
            _profile: &crate::config::Profile,
            messages: &[Msg],
            _tools: &[ToolSpec],
            temperature: Option<f32>,
            cancel: Arc<AtomicBool>,
            _on_delta: &mut dyn FnMut(Delta),
        ) -> Result<Completion> {
            let mut wires = self.wires.lock().unwrap();
            wires.push(messages.to_vec());
            self.temps.lock().unwrap().push(temperature);
            if self.cancel_on_call == Some(wires.len()) {
                cancel.store(true, Ordering::SeqCst);
            }
            self.replies
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| Err(Error::msg("test script exhausted")))
        }
    }

    fn completion(msg: Msg, finish_reason: &str) -> Completion {
        Completion { msg, finish_reason: finish_reason.into(), usage: None, timings: None }
    }

    fn done(text: &str) -> Result<Completion> {
        Ok(completion(
            Msg { role: "assistant".into(), content: Some(text.into()), tool_calls: None, tool_call_id: None, reasoning: None },
            "stop",
        ))
    }

    fn length(text: &str) -> Result<Completion> {
        Ok(completion(
            Msg { role: "assistant".into(), content: Some(text.into()), tool_calls: None, tool_call_id: None, reasoning: None },
            "length",
        ))
    }

    fn tool_calls(calls: &[(&str, &str)]) -> Result<Completion> {
        let tc = calls
            .iter()
            .enumerate()
            .map(|(i, (name, args))| ToolCall {
                id: format!("call_{i}"),
                kind: "function".into(),
                function: FunctionCall { name: (*name).into(), arguments: (*args).into() },
            })
            .collect();
        Ok(completion(
            Msg { role: "assistant".into(), content: None, tool_calls: Some(tc), tool_call_id: None, reasoning: None },
            "tool_calls",
        ))
    }

    fn with_usage(c: Result<Completion>, prompt_tokens: u64) -> Result<Completion> {
        c.map(|mut c| {
            c.usage = Some(Usage { prompt_tokens, completion_tokens: 0, total_tokens: prompt_tokens });
            c
        })
    }

    /// Agent in a unique temp cwd with a scripted provider; the dead
    /// base_url makes the background /props probe fail instantly.
    fn scripted_agent(
        replies: Vec<Result<Completion>>,
        cancel_on_call: Option<usize>,
    ) -> ScriptedTuple {
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "m-agent-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut config = Config::default();
        config.profile.base_url = "http://127.0.0.1:1".into();
        let mut agent = Agent::new(config, dir, "test system prompt".into(), Vec::new()).unwrap();
        let wires = Arc::new(Mutex::new(Vec::new()));
        let temps = Arc::new(Mutex::new(Vec::new()));
        agent.set_provider(Box::new(Scripted {
            replies: Mutex::new(replies.into()),
            wires: Arc::clone(&wires),
            temps: Arc::clone(&temps),
            cancel_on_call,
        }));
        (agent, wires, temps)
    }

    fn run(agent: &mut Agent, prompt: &str) -> (RunOutcome, Vec<AgentEvent>) {
        let mut events = Vec::new();
        let outcome = agent.run_prompt(prompt, &mut |e| events.push(e)).unwrap();
        (outcome, events)
    }

    fn tool_results(agent: &Agent) -> Vec<String> {
        agent
            .session
            .messages
            .iter()
            .filter(|m| m.role == "tool")
            .filter_map(|m| m.content.clone())
            .collect()
    }

    #[test]
    fn final_answer_without_tools() {
        let (mut agent, wires, _) = scripted_agent(vec![done("all done")], None);
        let (outcome, _) = run(&mut agent, "hi");
        assert_eq!(outcome.stop, StopReason::Done);
        assert_eq!(outcome.final_text, "all done");
        assert_eq!(outcome.turns, 1);
        // Wire = system prompt + user prompt.
        let wire = &wires.lock().unwrap()[0];
        assert_eq!(wire[0].role, "system");
        assert_eq!(wire.last().unwrap().content.as_deref(), Some("hi"));
    }

    #[test]
    fn length_runaway_is_discarded_and_nudged() {
        let (mut agent, wires, _) = scripted_agent(vec![length("RUNAWAY reasoning"), done("ok")], None);
        let (outcome, _) = run(&mut agent, "go");
        assert_eq!(outcome.stop, StopReason::Done);
        // The retry wire carries the corrective nudge but not the runaway text.
        let retry_wire = &wires.lock().unwrap()[1];
        let flat: String =
            retry_wire.iter().filter_map(|m| m.content.clone()).collect::<Vec<_>>().join("\n");
        assert!(flat.contains("overran the token limit"));
        assert!(!flat.contains("RUNAWAY"));
    }

    #[test]
    fn length_nudges_exhaust_to_length_stop() {
        let replies = (0..6).map(|_| length("x")).collect();
        let (mut agent, _, _) = scripted_agent(replies, None);
        let (outcome, _) = run(&mut agent, "go");
        assert_eq!(outcome.stop, StopReason::Length);
        assert_eq!(outcome.turns, 6);
        let nudges = agent
            .session
            .messages
            .iter()
            .filter(|m| {
                m.role == "user"
                    && m.content.as_deref().is_some_and(|c| c.contains("overran the token limit"))
            })
            .count();
        assert_eq!(nudges, 5);
        // The final truncated answer is kept.
        assert_eq!(agent.session.messages.last().unwrap().content.as_deref(), Some("x"));
    }

    #[test]
    fn repeated_identical_call_gets_escalating_note() {
        let echo = ("bash", r#"{"command":"echo stable"}"#);
        let (mut agent, _, _) =
            scripted_agent(vec![tool_calls(&[echo]), tool_calls(&[echo]), done("ok")], None);
        let (outcome, _) = run(&mut agent, "go");
        assert_eq!(outcome.stop, StopReason::Done);
        let results = tool_results(&agent);
        assert_eq!(results.len(), 2);
        assert!(!results[0].contains("(note:"));
        assert!(results[1].contains("run exactly this 2 times"));
    }

    #[test]
    fn successful_write_clears_repeat_tracking() {
        let echo = ("bash", r#"{"command":"echo stable"}"#);
        let write = ("write", r#"{"path":"f.txt","content":"x"}"#);
        let (mut agent, _, _) = scripted_agent(
            vec![tool_calls(&[echo]), tool_calls(&[write]), tool_calls(&[echo]), done("ok")],
            None,
        );
        let (outcome, _) = run(&mut agent, "go");
        assert_eq!(outcome.stop, StopReason::Done);
        let results = tool_results(&agent);
        assert_eq!(results.len(), 3);
        // The rerun after a state change is not annotated.
        assert!(!results[2].contains("(note:"), "got: {}", results[2]);
    }

    #[test]
    fn cancel_mid_turn_still_answers_every_tool_call() {
        let echo = ("bash", r#"{"command":"echo hi"}"#);
        let (mut agent, _, _) =
            scripted_agent(vec![tool_calls(&[echo, echo])], Some(1));
        let (outcome, _) = run(&mut agent, "go");
        assert_eq!(outcome.stop, StopReason::Cancelled);
        let results = tool_results(&agent);
        assert_eq!(results, vec!["Cancelled by user.", "Cancelled by user."]);
    }

    #[test]
    fn max_turns_stops_the_loop() {
        let echo = ("bash", r#"{"command":"echo hi"}"#);
        let (mut agent, _, _) =
            scripted_agent(vec![tool_calls(&[echo]), tool_calls(&[echo])], None);
        agent.config.max_turns = 2;
        let (outcome, _) = run(&mut agent, "go");
        assert_eq!(outcome.stop, StopReason::MaxTurns);
        assert_eq!(outcome.turns, 2);
    }

    #[test]
    fn steering_input_is_injected_after_tools() {
        let echo = ("bash", r#"{"command":"echo hi"}"#);
        let (mut agent, wires, _) = scripted_agent(vec![tool_calls(&[echo]), done("ok")], None);
        agent.steer.lock().unwrap().push_back("also check the docs".into());
        let (outcome, _) = run(&mut agent, "go");
        assert_eq!(outcome.stop, StopReason::Done);
        let retry_wire = &wires.lock().unwrap()[1];
        assert!(
            retry_wire
                .iter()
                .any(|m| m.role == "user" && m.content.as_deref() == Some("also check the docs"))
        );
    }

    #[test]
    fn transport_error_is_retried_once() {
        let (mut agent, wires, _) =
            scripted_agent(vec![Err(Error::msg("connection reset")), done("ok")], None);
        let (outcome, _) = run(&mut agent, "go");
        assert_eq!(outcome.stop, StopReason::Done);
        assert_eq!(wires.lock().unwrap().len(), 2);
    }

    #[test]
    fn api_error_is_not_retried() {
        let (mut agent, wires, _) =
            scripted_agent(vec![Err(Error::msg("API error (HTTP 400): bad request"))], None);
        let err = agent.run_prompt("go", &mut |_| {}).unwrap_err();
        assert!(err.to_string().contains("API error"));
        assert_eq!(wires.lock().unwrap().len(), 1);
    }

    #[test]
    fn context_guard_clips_memory_not_session_file() {
        let echo = ("bash", r#"{"command":"echo hi"}"#);
        let (mut agent, _, _) = scripted_agent(
            vec![with_usage(tool_calls(&[echo]), 900), done("ok")],
            None,
        );
        agent.ctx_limit_handle().store(1000, Ordering::Relaxed);
        let long = "y".repeat(700);
        for i in 0..12 {
            agent.session.push(Msg::tool_result(&format!("old_{i}"), &long)).unwrap();
        }
        let (outcome, events) = run(&mut agent, "go");
        assert_eq!(outcome.stop, StopReason::Done);
        assert!(events.iter().any(
            |e| matches!(e, AgentEvent::Notice(n) if n.contains("clipped"))
        ));
        let clipped_in_memory = agent
            .session
            .messages
            .iter()
            .filter(|m| m.content.as_deref().is_some_and(|c| c.contains("clipped to fit context")))
            .count();
        assert!(clipped_in_memory > 0);
        // The session file stays faithful.
        let reloaded = Session::load(&agent.session.path).unwrap();
        assert!(
            reloaded
                .messages
                .iter()
                .all(|m| !m.content.as_deref().unwrap_or("").contains("clipped to fit context"))
        );
        assert!(reloaded.messages.iter().any(|m| m.content.as_deref() == Some(long.as_str())));
    }

    #[test]
    fn recovery_temp_applies_on_length_nudge() {
        // A length-runaway should trigger temp 0.4 on the retry, then
        // fall back to the configured 0.0 once the model answers cleanly.
        let (mut agent, _, temps) = scripted_agent(vec![length("boom"), done("ok")], None);
        agent.config.profile.temperature = Some(0.0);
        let (outcome, _) = run(&mut agent, "go");
        assert_eq!(outcome.stop, StopReason::Done);
        let t: Vec<_> = temps.lock().unwrap().clone();
        assert_eq!(t, vec![Some(0.0), Some(0.4)], "temps: {t:?}");
    }

    #[test]
    fn recovery_temp_applies_on_identical_repeat() {
        let echo = ("bash", r#"{"command":"echo hi"}"#);
        // 3 identical tool calls with identical output → third triggers recovery
        let (mut agent, _, temps) = scripted_agent(
            vec![tool_calls(&[echo]), tool_calls(&[echo]), tool_calls(&[echo]), done("ok")],
            None,
        );
        agent.config.profile.temperature = Some(0.0);
        let (outcome, _) = run(&mut agent, "go");
        assert_eq!(outcome.stop, StopReason::Done);
        let t: Vec<_> = temps.lock().unwrap().clone();
        // 4 calls to the model: normal, normal, annotated-repeat→recovery, recovery (final answer)
        assert_eq!(t, vec![Some(0.0), Some(0.0), Some(0.4), Some(0.4)], "temps: {t:?}");
    }

    #[test]
    fn recovery_temp_unchanged_when_already_hot() {
        // When configured temp is already ≥ RECOVERY_TEMP, no change.
        let (mut agent, _, temps) = scripted_agent(vec![length("boom"), done("ok")], None);
        agent.config.profile.temperature = Some(0.7);
        let (outcome, _) = run(&mut agent, "go");
        assert_eq!(outcome.stop, StopReason::Done);
        let t: Vec<_> = temps.lock().unwrap().clone();
        assert_eq!(t, vec![Some(0.7), Some(0.7)], "temps: {t:?}");
    }

    #[test]
    fn recovery_temp_unchanged_when_server_default() {
        // Server default (None) should stay None.
        let (mut agent, _, temps) = scripted_agent(vec![length("boom"), done("ok")], None);
        agent.config.profile.temperature = None;
        let (outcome, _) = run(&mut agent, "go");
        assert_eq!(outcome.stop, StopReason::Done);
        let t: Vec<_> = temps.lock().unwrap().clone();
        assert_eq!(t, vec![None, None], "temps: {t:?}");
    }

    #[test]
    fn recovery_temp_fires_at_most_once_per_event() {
        // Two length runaways → recovery should fire for the first retry
        // only, then the second retry also gets recovery (since nudge is a
        // new recovery event).
        let (mut agent, _, temps) =
            scripted_agent(vec![length("a"), length("b"), done("ok")], None);
        agent.config.profile.temperature = Some(0.0);
        let (outcome, _) = run(&mut agent, "go");
        assert_eq!(outcome.stop, StopReason::Done);
        let t: Vec<_> = temps.lock().unwrap().clone();
        assert_eq!(t, vec![Some(0.0), Some(0.4), Some(0.4)], "temps: {t:?}");
    }
}
