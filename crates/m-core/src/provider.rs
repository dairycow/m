//! OpenAI-compatible chat completions with streaming, tool calls, and
//! llama.cpp telemetry (timings, cached tokens).

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::Profile;
use crate::error::{Error, Result};
use crate::http::{self, Url};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: FunctionCall,
}

fn function_type() -> String {
    "function".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments string, as on the wire.
    pub arguments: String,
}

/// One chat message, wire-compatible with the OpenAI API. `reasoning` is
/// kept locally (sessions, UI) and stripped from requests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Msg {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
}

impl Msg {
    pub fn system(content: impl Into<String>) -> Msg {
        Msg::plain("system", content)
    }
    pub fn user(content: impl Into<String>) -> Msg {
        Msg::plain("user", content)
    }
    pub fn tool_result(call_id: &str, content: impl Into<String>) -> Msg {
        Msg {
            role: "tool".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(call_id.to_string()),
            reasoning: None,
        }
    }
    fn plain(role: &str, content: impl Into<String>) -> Msg {
        Msg {
            role: role.into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
        }
    }

    fn to_wire(&self) -> Value {
        let mut v = json!({ "role": self.role });
        // OpenAI requires content present (possibly "") except when the
        // assistant message only carries tool calls.
        v["content"] = json!(self.content.clone().unwrap_or_default());
        if let Some(tc) = &self.tool_calls {
            v["tool_calls"] = serde_json::to_value(tc).unwrap_or(Value::Null);
        }
        if let Some(id) = &self.tool_call_id {
            v["tool_call_id"] = json!(id);
        }
        v
    }
}

/// Tool definition sent to the model.
#[derive(Debug, Clone, Serialize)]
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: Value,
}

impl ToolSpec {
    fn to_wire(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }
}

/// llama.cpp per-response timings (absent on other providers).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Timings {
    #[serde(default)]
    pub prompt_n: u64,
    #[serde(default)]
    pub prompt_per_second: f64,
    #[serde(default)]
    pub predicted_n: u64,
    #[serde(default)]
    pub predicted_per_second: f64,
    #[serde(default)]
    pub draft_n: u64,
    #[serde(default)]
    pub draft_n_accepted: u64,
    #[serde(default)]
    pub cache_n: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct Completion {
    pub msg: Msg,
    pub finish_reason: String,
    pub usage: Option<Usage>,
    pub timings: Option<Timings>,
}

/// Streaming deltas surfaced to the UI while a response is in flight.
#[derive(Debug, Clone)]
pub enum Delta {
    Reasoning(String),
    Content(String),
    /// A tool call's name became known (arguments still streaming).
    ToolCallBegin {
        index: usize,
        name: String,
    },
}

/// The chat endpoint as a seam: the agent loop talks to this trait, so
/// tests can drive it with scripted completions (no server, no GPU) and
/// alternative transports stay possible. The default is [`Http`], the
/// hand-rolled SSE client below.
pub trait ChatProvider: Send {
    fn stream_chat(
        &self,
        profile: &Profile,
        messages: &[Msg],
        tools: &[ToolSpec],
        temperature: Option<f32>,
        cancel: Arc<AtomicBool>,
        on_delta: &mut dyn FnMut(Delta),
    ) -> Result<Completion>;
}

/// Default provider: streaming HTTP against the configured base_url.
pub struct Http;

impl ChatProvider for Http {
    fn stream_chat(
        &self,
        profile: &Profile,
        messages: &[Msg],
        tools: &[ToolSpec],
        temperature: Option<f32>,
        cancel: Arc<AtomicBool>,
        on_delta: &mut dyn FnMut(Delta),
    ) -> Result<Completion> {
        stream_chat(profile, messages, tools, temperature, cancel, on_delta)
    }
}

#[derive(Default)]
struct ToolCallAcc {
    id: String,
    name: String,
    arguments: String,
    announced: bool,
}

/// Endpoint path to append to the base_url. Bases that already carry an
/// API version segment (…/api/v1, …/coding/paas/v4) get "/chat/completions";
/// bare hosts like the local llama-server get the conventional
/// "/v1/chat/completions".
fn chat_path(base_url: &str) -> &'static str {
    let last = base_url
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or("");
    let versioned =
        last.len() >= 2 && last.starts_with('v') && last[1..].bytes().all(|b| b.is_ascii_digit());
    if versioned {
        "/chat/completions"
    } else {
        "/v1/chat/completions"
    }
}

/// POST a streaming chat completion. `on_delta` fires per streamed fragment;
/// the assembled message is returned at the end.
pub fn stream_chat(
    profile: &Profile,
    messages: &[Msg],
    tools: &[ToolSpec],
    temperature: Option<f32>,
    cancel: Arc<AtomicBool>,
    mut on_delta: impl FnMut(Delta),
) -> Result<Completion> {
    let url = Url::join(&profile.base_url, chat_path(&profile.base_url))?;
    let mut body = json!({
        "model": profile.model,
        "messages": messages.iter().map(Msg::to_wire).collect::<Vec<_>>(),
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools.iter().map(ToolSpec::to_wire).collect());
    }
    if let Some(t) = temperature.or(profile.temperature) {
        body["temperature"] = json!(t);
    }
    if let Some(m) = profile.max_tokens {
        body["max_tokens"] = json!(m);
    }

    let auth = format!("Bearer {}", profile.api_key);
    let headers: Vec<(&str, &str)> = if profile.api_key.is_empty() {
        vec![]
    } else {
        vec![("Authorization", &auth)]
    };
    let body_bytes = serde_json::to_vec(&body)?;
    let mut resp = http::post_json(&url, &headers, &body_bytes, cancel)?;

    if resp.status >= 400 {
        let text = resp.read_to_string().unwrap_or_default();
        let msg = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| {
                v.pointer("/error/message")
                    .and_then(|m| m.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_else(|| http::truncate(&text, 400));
        return Err(Error::msg(format!(
            "API error (HTTP {}): {}",
            resp.status, msg
        )));
    }

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_accs: Vec<ToolCallAcc> = Vec::new();
    let mut finish_reason = String::new();
    let mut usage: Option<Usage> = None;
    let mut timings: Option<Timings> = None;

    while let Some(line) = resp.next_line()? {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        let chunk: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue, // tolerate keep-alive noise
        };
        if let Some(u) = chunk.get("usage").filter(|u| !u.is_null()) {
            usage = serde_json::from_value(u.clone()).ok();
        }
        if let Some(t) = chunk.get("timings").filter(|t| !t.is_null()) {
            timings = serde_json::from_value(t.clone()).ok();
        }
        let Some(choice) = chunk.pointer("/choices/0") else {
            continue;
        };
        if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            finish_reason = fr.to_string();
        }
        let Some(delta) = choice.get("delta") else {
            continue;
        };
        if let Some(s) = delta.get("reasoning_content").and_then(|v| v.as_str())
            && !s.is_empty()
        {
            reasoning.push_str(s);
            on_delta(Delta::Reasoning(s.to_string()));
        }
        if let Some(s) = delta.get("content").and_then(|v| v.as_str())
            && !s.is_empty()
        {
            content.push_str(s);
            on_delta(Delta::Content(s.to_string()));
        }
        if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for call in calls {
                let index = call.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                while tool_accs.len() <= index {
                    tool_accs.push(ToolCallAcc::default());
                }
                let acc = &mut tool_accs[index];
                if let Some(id) = call.get("id").and_then(|v| v.as_str()) {
                    acc.id.push_str(id);
                }
                if let Some(name) = call.pointer("/function/name").and_then(|v| v.as_str()) {
                    acc.name.push_str(name);
                }
                if let Some(args) = call.pointer("/function/arguments").and_then(|v| v.as_str()) {
                    acc.arguments.push_str(args);
                }
                if !acc.announced && !acc.name.is_empty() {
                    acc.announced = true;
                    on_delta(Delta::ToolCallBegin {
                        index,
                        name: acc.name.clone(),
                    });
                }
            }
        }
    }

    let tool_calls: Vec<ToolCall> = tool_accs
        .into_iter()
        .filter(|a| !a.name.is_empty())
        .enumerate()
        .map(|(i, a)| ToolCall {
            id: if a.id.is_empty() {
                format!("call_{i}")
            } else {
                a.id
            },
            kind: "function".into(),
            function: FunctionCall {
                name: a.name,
                arguments: a.arguments,
            },
        })
        .collect();

    let msg = Msg {
        role: "assistant".into(),
        content: if content.is_empty() && !tool_calls.is_empty() {
            None
        } else {
            Some(content)
        },
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        tool_call_id: None,
        reasoning: if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        },
    };
    Ok(Completion {
        msg,
        finish_reason,
        usage,
        timings,
    })
}

/// Query llama-server's /props for the true context size. Returns None for
/// providers without the endpoint.
pub fn probe_ctx(profile: &Profile) -> Option<usize> {
    let url = Url::join(&profile.base_url, "/props").ok()?;
    let cancel = Arc::new(AtomicBool::new(false));
    let body = http::get_json(&url, &[], cancel).ok()?;
    let v: Value = serde_json::from_str(&body).ok()?;
    v.pointer("/default_generation_settings/n_ctx")
        .and_then(|n| n.as_u64())
        .map(|n| n as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_path_respects_versioned_bases() {
        assert_eq!(chat_path("http://localhost:8080"), "/v1/chat/completions");
        assert_eq!(
            chat_path("https://openrouter.ai/api/v1"),
            "/chat/completions"
        );
        assert_eq!(
            chat_path("https://openrouter.ai/api/v1/"),
            "/chat/completions"
        );
        assert_eq!(
            chat_path("https://api.z.ai/api/coding/paas/v4/"),
            "/chat/completions"
        );
        // "v" followed by non-digits is not a version segment.
        assert_eq!(
            chat_path("https://example.com/api/vein"),
            "/v1/chat/completions"
        );
    }
}
