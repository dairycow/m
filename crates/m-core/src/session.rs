//! Append-only JSONL sessions under ~/.local/share/m/sessions/<cwd-hash>/.
//! The file is a faithful log; context-window truncation is applied only to
//! the in-memory copy sent to the model.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::provider::Msg;

#[derive(Debug, Serialize, Deserialize)]
struct Header {
    version: u32,
    cwd: String,
    created_unix: u64,
    model: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Line {
    Header(Header),
    Msg { msg: Msg },
}

pub struct Session {
    pub id: String,
    pub path: PathBuf,
    pub messages: Vec<Msg>,
}

fn cwd_key(cwd: &Path) -> String {
    // Stable, readable directory key: last component + short hash.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in cwd.to_string_lossy().bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    let name = cwd.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    format!("{name}-{h:012x}")
}

fn sessions_dir(cwd: &Path) -> PathBuf {
    crate::config::data_dir().join("sessions").join(cwd_key(cwd))
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Session {
    pub fn new(cwd: &Path, model: &str) -> Result<Session> {
        let dir = sessions_dir(cwd);
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::msg(format!("create {}: {e}", dir.display())))?;
        let id = format!("{}-{:04x}", now_unix(), std::process::id() & 0xffff);
        let path = dir.join(format!("{id}.jsonl"));
        let header = Line::Header(Header {
            version: 1,
            cwd: cwd.to_string_lossy().into_owned(),
            created_unix: now_unix(),
            model: model.to_string(),
        });
        append_line(&path, &header)?;
        Ok(Session { id, path, messages: Vec::new() })
    }

    /// Most recently modified session for this cwd, if any.
    pub fn latest(cwd: &Path) -> Option<PathBuf> {
        let dir = sessions_dir(cwd);
        let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
        for entry in std::fs::read_dir(dir).ok()?.flatten() {
            let p = entry.path();
            if p.extension().is_some_and(|e| e == "jsonl")
                && let Ok(meta) = entry.metadata()
                && let Ok(mtime) = meta.modified()
                && best.as_ref().is_none_or(|(t, _)| mtime > *t)
            {
                best = Some((mtime, p));
            }
        }
        best.map(|(_, p)| p)
    }

    /// All sessions for this cwd, newest first: (path, created, first user msg).
    pub fn list(cwd: &Path) -> Vec<(PathBuf, u64, String)> {
        let dir = sessions_dir(cwd);
        let mut out = Vec::new();
        let Ok(entries) = std::fs::read_dir(dir) else { return out };
        for entry in entries.flatten() {
            let p = entry.path();
            if !p.extension().is_some_and(|e| e == "jsonl") {
                continue;
            }
            if let Ok(s) = Session::load(&p) {
                let created = p
                    .file_stem()
                    .and_then(|s| s.to_string_lossy().split('-').next().map(String::from))
                    .and_then(|t| t.parse().ok())
                    .unwrap_or(0);
                let first = s
                    .messages
                    .iter()
                    .find(|m| m.role == "user")
                    .and_then(|m| m.content.clone())
                    .unwrap_or_default();
                out.push((p, created, crate::http::truncate(first.trim(), 80)));
            }
        }
        out.sort_by_key(|e| std::cmp::Reverse(e.1));
        out
    }

    pub fn load(path: &Path) -> Result<Session> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
        let mut messages = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Line>(line) {
                Ok(Line::Msg { msg }) => messages.push(msg),
                Ok(Line::Header(_)) => {}
                Err(_) => {} // tolerate partial trailing writes
            }
        }
        let id = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        Ok(Session { id, path: path.to_path_buf(), messages })
    }

    pub fn push(&mut self, msg: Msg) -> Result<()> {
        append_line(&self.path, &Line::Msg { msg: msg.clone() })?;
        self.messages.push(msg);
        Ok(())
    }
}

fn append_line(path: &Path, line: &Line) -> Result<()> {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
    let mut s = serde_json::to_string(line)?;
    s.push('\n');
    f.write_all(s.as_bytes()).map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join(format!("m-sess-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Point data dir at tmp via direct path use: create in real data dir
        // is fine for test isolation since cwd is unique.
        let mut s = Session::new(&dir, "test-model").unwrap();
        s.push(Msg::user("hello")).unwrap();
        s.push(Msg {
            role: "assistant".into(),
            content: Some("hi".into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning: Some("thinking...".into()),
        })
        .unwrap();
        let loaded = Session::load(&s.path).unwrap();
        assert_eq!(loaded.messages, s.messages);
        assert_eq!(loaded.messages[1].reasoning.as_deref(), Some("thinking..."));
        std::fs::remove_file(&s.path).ok();
    }
}
