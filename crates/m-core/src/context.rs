//! File-based extensibility: hierarchical AGENTS.md project context and
//! SKILL.md discovery (Claude Code-compatible layout). Zero baseline cost:
//! nothing is loaded unless the files exist, and skills contribute only a
//! one-line index entry until the model asks for one.

use std::path::{Path, PathBuf};

const CONTEXT_CAP: usize = 24 * 1024;

/// Concatenate global + repo-hierarchy AGENTS.md (CLAUDE.md as fallback).
pub fn load_project_context(cwd: &Path) -> String {
    let mut parts: Vec<(PathBuf, String)> = Vec::new();
    let global = crate::config::config_dir().join("AGENTS.md");
    if let Ok(s) = std::fs::read_to_string(&global) {
        parts.push((global, s));
    }
    // Walk from the filesystem root down to cwd so outer (more general)
    // context comes first.
    let dirs: Vec<&Path> = {
        let mut v: Vec<&Path> = cwd.ancestors().collect();
        v.reverse();
        v
    };
    for dir in dirs {
        for name in ["AGENTS.md", "CLAUDE.md"] {
            let p = dir.join(name);
            if let Ok(s) = std::fs::read_to_string(&p) {
                parts.push((p, s));
                break; // AGENTS.md wins over CLAUDE.md in the same dir
            }
        }
    }
    let mut out = String::new();
    for (path, text) in parts {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&format!("<!-- {} -->\n", path.display()));
        out.push_str(text.trim());
        out.push('\n');
        if out.len() > CONTEXT_CAP {
            out.truncate(CONTEXT_CAP);
            out.push_str("\n(project context truncated)\n");
            break;
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct SkillInfo {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
}

/// Scan skill directories, most specific last (later wins on name clash):
/// ~/.claude/skills, ~/.config/m/skills, <cwd>/.claude/skills, <cwd>/.m/skills.
pub fn discover_skills(cwd: &Path) -> Vec<SkillInfo> {
    let home = dirs::home_dir().unwrap_or_default();
    let roots = [
        home.join(".claude/skills"),
        crate::config::config_dir().join("skills"),
        cwd.join(".claude/skills"),
        cwd.join(".m/skills"),
    ];
    let mut by_name: std::collections::BTreeMap<String, SkillInfo> = Default::default();
    for root in roots {
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        for entry in entries.flatten() {
            let skill_md = entry.path().join("SKILL.md");
            let Ok(head) = read_head(&skill_md, 4096) else {
                continue;
            };
            let fm = frontmatter(&head);
            let dir_name = entry.file_name().to_string_lossy().into_owned();
            let name = fm.get("name").cloned().unwrap_or(dir_name);
            let description = fm.get("description").cloned().unwrap_or_default();
            by_name.insert(
                name.clone(),
                SkillInfo {
                    name,
                    description: crate::http::truncate(&description, 200),
                    path: skill_md,
                },
            );
        }
    }
    by_name.into_values().collect()
}

/// One line per skill for the system prompt.
pub fn skills_index(skills: &[SkillInfo]) -> String {
    let mut out = String::new();
    for s in skills {
        out.push_str(&format!("- {}: {}\n", s.name, s.description));
    }
    out
}

fn read_head(path: &Path, max: usize) -> std::io::Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; max];
    let n = f.read(&mut buf)?;
    buf.truncate(n);
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Minimal YAML frontmatter: `key: value` pairs between --- fences.
/// Handles quoted values and simple multi-line folded text by ignoring it.
fn frontmatter(text: &str) -> std::collections::BTreeMap<String, String> {
    let mut map = std::collections::BTreeMap::new();
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("---") {
        return map;
    }
    for line in lines {
        if line.trim() == "---" {
            break;
        }
        if line.starts_with(' ') || line.starts_with('\t') {
            continue; // nested/multi-line values: skip
        }
        if let Some((k, v)) = line.split_once(':') {
            let v = v.trim().trim_matches('"').trim_matches('\'').to_string();
            map.insert(k.trim().to_string(), v);
        }
    }
    map
}

/// Slash-command templates: ~/.config/m/commands/*.md and <cwd>/.m/commands/*.md.
#[derive(Debug, Clone)]
pub struct CommandTemplate {
    /// Without the leading slash.
    pub name: String,
    pub description: String,
    pub path: PathBuf,
}

pub fn discover_commands(cwd: &Path) -> Vec<CommandTemplate> {
    let roots = [
        crate::config::config_dir().join("commands"),
        cwd.join(".m/commands"),
    ];
    let mut by_name: std::collections::BTreeMap<String, CommandTemplate> = Default::default();
    for root in roots {
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().is_none_or(|e| e != "md") {
                continue;
            }
            let Some(name) = p.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
                continue;
            };
            let head = read_head(&p, 2048).unwrap_or_default();
            let fm = frontmatter(&head);
            let description = fm.get("description").cloned().unwrap_or_else(|| {
                strip_frontmatter(&head)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_string()
            });
            by_name.insert(
                name.clone(),
                CommandTemplate {
                    name,
                    description: crate::http::truncate(&description, 80),
                    path: p,
                },
            );
        }
    }
    by_name.into_values().collect()
}

/// Expand a command template with `$ARGUMENTS` substitution.
pub fn expand_command(template_path: &Path, args: &str) -> std::io::Result<String> {
    let text = std::fs::read_to_string(template_path)?;
    let body = strip_frontmatter(&text);
    if body.contains("$ARGUMENTS") {
        Ok(body.replace("$ARGUMENTS", args))
    } else if args.is_empty() {
        Ok(body.to_string())
    } else {
        Ok(format!("{body}\n\n{args}"))
    }
}

fn strip_frontmatter(text: &str) -> &str {
    let t = text.trim_start();
    if let Some(rest) = t.strip_prefix("---")
        && let Some(end) = rest.find("\n---")
    {
        return rest[end + 4..].trim_start_matches(['\n', '\r']);
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontmatter_parse() {
        let fm = frontmatter("---\nname: foo\ndescription: \"does things\"\n---\nbody");
        assert_eq!(fm.get("name").map(String::as_str), Some("foo"));
        assert_eq!(
            fm.get("description").map(String::as_str),
            Some("does things")
        );
        assert!(frontmatter("no fence").is_empty());
    }

    #[test]
    fn strip_and_expand() {
        assert_eq!(strip_frontmatter("---\na: b\n---\nhello"), "hello");
        assert_eq!(strip_frontmatter("plain"), "plain");
        let dir = std::env::temp_dir().join(format!("m-cmd-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.md");
        std::fs::write(&p, "---\ndescription: test\n---\nDo $ARGUMENTS now").unwrap();
        assert_eq!(expand_command(&p, "the thing").unwrap(), "Do the thing now");
    }
}
