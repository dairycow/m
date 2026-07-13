//! The four core tools: read, write, edit, bash. pi's set — sufficient
//! because models already know the coding-agent pattern.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::context::SkillInfo;
use crate::error::{Error, Result};
use crate::provider::ToolSpec;

const READ_MAX_BYTES: usize = 256 * 1024;
const READ_MAX_LINES: usize = 2000;
const BASH_MAX_OUTPUT: usize = 48 * 1024;
const BASH_DEFAULT_TIMEOUT: u64 = 120;
const BASH_MAX_TIMEOUT: u64 = 900;

pub fn specs(with_skill_tool: bool) -> Vec<ToolSpec> {
    let mut specs = vec![
        ToolSpec {
            name: "read",
            description: "Read a file. Returns at most 2000 lines starting at `offset` (1-based line number).",
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path (absolute or relative to the working directory)" },
                    "offset": { "type": "integer", "description": "1-based line to start from (default 1)" },
                    "limit": { "type": "integer", "description": "Max lines to return (default 2000)" }
                },
                "required": ["path"]
            }),
        },
        ToolSpec {
            name: "write",
            description: "Create or overwrite a file with the given content. Creates parent directories.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path" },
                    "content": { "type": "string", "description": "Full file content" }
                },
                "required": ["path", "content"]
            }),
        },
        ToolSpec {
            name: "edit",
            description: "Replace text in a file. `old_string` must match the file contents exactly and (unless `replace_all`) uniquely; include enough surrounding lines to make it unique.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path" },
                    "old_string": { "type": "string", "description": "Exact text to replace" },
                    "new_string": { "type": "string", "description": "Replacement text" },
                    "replace_all": { "type": "boolean", "description": "Replace every occurrence (default false)" }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        },
        ToolSpec {
            name: "bash",
            description: "Run a shell command (bash -c) in the working directory and return its combined output. Not interactive. `timeout` in seconds, default 120, max 900.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "Command to run" },
                    "timeout": { "type": "integer", "description": "Seconds before the command is killed" }
                },
                "required": ["command"]
            }),
        },
    ];
    if with_skill_tool {
        specs.push(ToolSpec {
            name: "skill",
            description: "Load the full instructions of a skill by name (skills are listed in the system prompt). Use when the current task matches a skill's description.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Skill name" }
                },
                "required": ["name"]
            }),
        });
    }
    specs
}

/// The outcome of a tool execution; `is_error` results are still returned to
/// the model so it can self-correct. `detail` is UI-only (e.g. a diff for
/// edit/write) and never sent to the model.
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
    pub detail: Option<String>,
}

impl ToolOutput {
    fn ok(content: impl Into<String>) -> ToolOutput {
        ToolOutput {
            content: content.into(),
            is_error: false,
            detail: None,
        }
    }
    fn err(content: impl Into<String>) -> ToolOutput {
        ToolOutput {
            content: content.into(),
            is_error: true,
            detail: None,
        }
    }
    fn with_detail(mut self, detail: String) -> ToolOutput {
        self.detail = Some(detail);
        self
    }
}

/// Unified-style diff for UI display, capped in size.
fn diff_text(old: &str, new: &str) -> String {
    let diff = similar::TextDiff::from_lines(old, new);
    let mut out = String::new();
    for hunk in diff.unified_diff().context_radius(2).iter_hunks() {
        for change in hunk.iter_changes() {
            let sign = match change.tag() {
                similar::ChangeTag::Delete => "-",
                similar::ChangeTag::Insert => "+",
                similar::ChangeTag::Equal => " ",
            };
            out.push_str(sign);
            out.push_str(change.value().trim_end_matches('\n'));
            out.push('\n');
            if out.len() > 16 * 1024 {
                out.push_str("(… diff truncated …)\n");
                return out;
            }
        }
    }
    out
}

pub fn execute(
    name: &str,
    arguments: &str,
    cwd: &Path,
    skills: &[SkillInfo],
    cancel: &Arc<AtomicBool>,
) -> ToolOutput {
    let args: Value = match serde_json::from_str(arguments) {
        Ok(v) => v,
        Err(e) => {
            return ToolOutput::err(format!(
                "Invalid JSON arguments for tool '{name}': {e}. Arguments were: {}",
                crate::http::truncate(arguments, 400)
            ));
        }
    };
    let res = match name {
        "read" => read_tool(&args, cwd),
        "write" => write_tool(&args, cwd),
        "edit" => edit_tool(&args, cwd),
        "bash" => bash_tool(&args, cwd, cancel),
        "skill" => skill_tool(&args, skills),
        _ => Err(Error::msg(format!("Unknown tool: {name}"))),
    };
    match res {
        Ok(out) => out,
        Err(Error::Cancelled) => ToolOutput::err("Cancelled by user."),
        Err(e) => ToolOutput::err(format!("Error: {e}")),
    }
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::msg(format!("missing required string argument '{key}'")))
}

fn resolve(path: &str, cwd: &Path) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    }
}

fn read_tool(args: &Value, cwd: &Path) -> Result<ToolOutput> {
    let path = resolve(str_arg(args, "path")?, cwd);
    let offset = args
        .get("offset")
        .and_then(|v| v.as_u64())
        .unwrap_or(1)
        .max(1) as usize;
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(READ_MAX_LINES)
        .min(READ_MAX_LINES);

    let bytes = std::fs::read(&path).map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
    if bytes.iter().take(8192).any(|&b| b == 0) {
        return Ok(ToolOutput::ok(format!(
            "(binary file, {} bytes — not shown)",
            bytes.len()
        )));
    }
    let text = String::from_utf8_lossy(&bytes);
    let total_lines = text.lines().count();
    let mut out = String::new();
    let mut shown = 0usize;
    for line in text.lines().skip(offset - 1).take(limit) {
        out.push_str(line);
        out.push('\n');
        shown += 1;
        if out.len() > READ_MAX_BYTES {
            break;
        }
    }
    let last = offset - 1 + shown;
    if shown == 0 {
        return Ok(ToolOutput::ok(format!(
            "(file has {total_lines} lines; offset {offset} is past the end)"
        )));
    }
    if last < total_lines {
        out.push_str(&format!(
            "\n(showing lines {offset}-{last} of {total_lines}; continue with offset={})\n",
            last + 1
        ));
    }
    Ok(ToolOutput::ok(out))
}

fn write_tool(args: &Value, cwd: &Path) -> Result<ToolOutput> {
    let path = resolve(str_arg(args, "path")?, cwd);
    let content = str_arg(args, "content")?;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| Error::msg(format!("create {}: {e}", parent.display())))?;
    }
    let old = std::fs::read_to_string(&path).ok();
    std::fs::write(&path, content).map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
    let detail = diff_text(old.as_deref().unwrap_or(""), content);
    Ok(ToolOutput::ok(format!(
        "{} {} ({} bytes)",
        if old.is_some() { "Overwrote" } else { "Wrote" },
        path.display(),
        content.len()
    ))
    .with_detail(detail))
}

fn edit_tool(args: &Value, cwd: &Path) -> Result<ToolOutput> {
    let path = resolve(str_arg(args, "path")?, cwd);
    let old = str_arg(args, "old_string")?;
    let new = str_arg(args, "new_string")?;
    let replace_all = args
        .get("replace_all")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if old.is_empty() {
        return Err(Error::msg("old_string must not be empty"));
    }
    if old == new {
        return Err(Error::msg("old_string and new_string are identical"));
    }
    let text = std::fs::read_to_string(&path)
        .map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
    let count = text.matches(old).count();
    match count {
        0 => Err(Error::msg(format!(
            "old_string not found in {}. Re-read the file; the text must match exactly (including whitespace).",
            path.display()
        ))),
        1 => {
            let updated = text.replacen(old, new, 1);
            std::fs::write(&path, &updated)
                .map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
            Ok(ToolOutput::ok(format!("Edited {}", path.display()))
                .with_detail(diff_text(&text, &updated)))
        }
        n if replace_all => {
            let updated = text.replace(old, new);
            std::fs::write(&path, &updated)
                .map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
            Ok(
                ToolOutput::ok(format!("Edited {} ({n} replacements)", path.display()))
                    .with_detail(diff_text(&text, &updated)),
            )
        }
        n => Err(Error::msg(format!(
            "old_string occurs {n} times in {}. Add surrounding context to make it unique, or set replace_all=true.",
            path.display()
        ))),
    }
}

fn skill_tool(args: &Value, skills: &[SkillInfo]) -> Result<ToolOutput> {
    let name = str_arg(args, "name")?;
    let Some(skill) = skills.iter().find(|s| s.name == name) else {
        let available = skills
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(Error::msg(format!(
            "no skill named '{name}' (available: {available})"
        )));
    };
    let text = std::fs::read_to_string(&skill.path)
        .map_err(|e| Error::msg(format!("{}: {e}", skill.path.display())))?;
    let dir = skill
        .path
        .parent()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    Ok(ToolOutput::ok(format!(
        "(skill '{name}', files in {dir})\n\n{text}"
    )))
}

fn bash_tool(args: &Value, cwd: &Path, cancel: &Arc<AtomicBool>) -> Result<ToolOutput> {
    let command = str_arg(args, "command")?;
    let timeout = args
        .get("timeout")
        .and_then(|v| v.as_u64())
        .unwrap_or(BASH_DEFAULT_TIMEOUT)
        .min(BASH_MAX_TIMEOUT);
    run_bash(command, cwd, Duration::from_secs(timeout), cancel)
}

/// Run a command in its own process group, merge stdout/stderr, kill the
/// whole group on timeout or cancel.
pub fn run_bash(
    command: &str,
    cwd: &Path,
    timeout: Duration,
    cancel: &Arc<AtomicBool>,
) -> Result<ToolOutput> {
    use std::os::unix::process::CommandExt;

    let mut child = Command::new("bash")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("GIT_PAGER", "cat")
        .env("PAGER", "cat")
        .env("TERM", "dumb")
        .env("NO_COLOR", "1")
        .process_group(0)
        .spawn()
        .map_err(|e| Error::msg(format!("spawn bash: {e}")))?;

    let pgid = child.id() as i32;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    // Drain both pipes on threads to avoid deadlock on full buffers.
    let out_handle = std::thread::spawn(move || drain(stdout));
    let err_handle = std::thread::spawn(move || drain(stderr));

    let start = Instant::now();
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if cancel.load(Ordering::Relaxed) {
                    kill_group(pgid);
                    child.wait().ok();
                    break None;
                }
                if start.elapsed() > timeout {
                    timed_out = true;
                    kill_group(pgid);
                    break child.wait().ok();
                }
                std::thread::sleep(Duration::from_millis(15));
            }
            Err(e) => return Err(Error::msg(format!("wait: {e}"))),
        }
    };

    let mut output = out_handle.join().unwrap_or_default();
    let err_out = err_handle.join().unwrap_or_default();
    if !err_out.is_empty() {
        if !output.is_empty() && !output.ends_with('\n') {
            output.push('\n');
        }
        output.push_str(&err_out);
    }
    let output = clip_output(&output);

    let Some(status) = status else {
        return Err(Error::Cancelled);
    };
    if timed_out {
        return Ok(ToolOutput::err(format!(
            "Command timed out after {}s and was killed.\n{output}",
            timeout.as_secs()
        )));
    }
    match status.code() {
        Some(0) => Ok(ToolOutput::ok(if output.is_empty() {
            "(no output, exit 0)".to_string()
        } else {
            output
        })),
        Some(code) => Ok(ToolOutput::err(format!("Exit code {code}\n{output}"))),
        None => Ok(ToolOutput::err(format!("Killed by signal\n{output}"))),
    }
}

fn drain(mut pipe: impl std::io::Read) -> String {
    let mut buf = Vec::new();
    // Cap what we keep in memory at ~4x the clip size (head+tail live inside).
    let mut chunk = [0u8; 8192];
    let mut truncated_mid = 0usize;
    loop {
        match pipe.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if buf.len() > BASH_MAX_OUTPUT * 4 {
                    // Keep head and tail, drop the middle.
                    let keep = BASH_MAX_OUTPUT * 2;
                    let tail_start = buf.len() - keep;
                    truncated_mid += buf.len() - keep * 2;
                    let tail = buf.split_off(tail_start);
                    buf.truncate(keep);
                    buf.extend_from_slice(&tail);
                }
            }
        }
    }
    let mut s = String::from_utf8_lossy(&buf).into_owned();
    if truncated_mid > 0 {
        s.push_str(&format!("\n(… {truncated_mid} bytes of output dropped …)"));
    }
    s
}

fn clip_output(s: &str) -> String {
    if s.len() <= BASH_MAX_OUTPUT {
        return s.trim_end().to_string();
    }
    let head_end = floor_char(s, BASH_MAX_OUTPUT / 2);
    let tail_start = ceil_char(s, s.len() - BASH_MAX_OUTPUT / 2);
    format!(
        "{}\n(… {} bytes truncated …)\n{}",
        &s[..head_end],
        tail_start - head_end,
        s[tail_start..].trim_end()
    )
}

fn floor_char(s: &str, mut i: usize) -> usize {
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char(s: &str, mut i: usize) -> usize {
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

fn kill_group(pgid: i32) {
    unsafe {
        libc::kill(-pgid, libc::SIGKILL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir() -> PathBuf {
        let d = std::env::temp_dir().join(format!("m-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn cancel() -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(false))
    }

    #[test]
    fn edit_unique_and_ambiguous() {
        let dir = tmpdir();
        let f = dir.join("edit.txt");
        std::fs::write(&f, "aaa\nbbb\naaa\n").unwrap();
        let path = f.to_str().unwrap();

        // Ambiguous
        let out = execute(
            "edit",
            &json!({"path": path, "old_string": "aaa", "new_string": "ccc"}).to_string(),
            &dir,
            &[],
            &cancel(),
        );
        assert!(out.is_error, "{}", out.content);
        assert!(out.content.contains("2 times"));

        // Unique
        let out = execute(
            "edit",
            &json!({"path": path, "old_string": "bbb", "new_string": "BBB"}).to_string(),
            &dir,
            &[],
            &cancel(),
        );
        assert!(!out.is_error, "{}", out.content);
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "aaa\nBBB\naaa\n");

        // replace_all
        let out = execute(
            "edit",
            &json!({"path": path, "old_string": "aaa", "new_string": "ccc", "replace_all": true})
                .to_string(),
            &dir,
            &[],
            &cancel(),
        );
        assert!(!out.is_error);
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "ccc\nBBB\nccc\n");

        // Not found
        let out = execute(
            "edit",
            &json!({"path": path, "old_string": "zzz", "new_string": "y"}).to_string(),
            &dir,
            &[],
            &cancel(),
        );
        assert!(out.is_error);
        assert!(out.content.contains("not found"));
    }

    #[test]
    fn read_offset_limit() {
        let dir = tmpdir();
        let f = dir.join("read.txt");
        let body: String = (1..=100).map(|i| format!("line{i}\n")).collect();
        std::fs::write(&f, &body).unwrap();
        let out = execute(
            "read",
            &json!({"path": f.to_str().unwrap(), "offset": 50, "limit": 2}).to_string(),
            &dir,
            &[],
            &cancel(),
        );
        assert!(!out.is_error);
        assert!(out.content.starts_with("line50\nline51\n"));
        assert!(out.content.contains("offset=52"));
    }

    #[test]
    fn write_creates_dirs() {
        let dir = tmpdir();
        let f = dir.join("sub/dir/new.txt");
        let out = execute(
            "write",
            &json!({"path": f.to_str().unwrap(), "content": "hi"}).to_string(),
            &dir,
            &[],
            &cancel(),
        );
        assert!(!out.is_error, "{}", out.content);
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "hi");
    }

    #[test]
    fn bash_exit_codes_and_timeout() {
        let dir = tmpdir();
        let out = execute(
            "bash",
            &json!({"command": "echo hi; echo err >&2"}).to_string(),
            &dir,
            &[],
            &cancel(),
        );
        assert!(!out.is_error);
        assert!(out.content.contains("hi") && out.content.contains("err"));

        let out = execute(
            "bash",
            &json!({"command": "exit 3"}).to_string(),
            &dir,
            &[],
            &cancel(),
        );
        assert!(out.is_error);
        assert!(out.content.contains("Exit code 3"));

        let out = execute(
            "bash",
            &json!({"command": "sleep 30", "timeout": 1}).to_string(),
            &dir,
            &[],
            &cancel(),
        );
        assert!(out.is_error);
        assert!(out.content.contains("timed out"));
    }
}
