//! Project file listing for the `@` file picker: `git ls-files` when `cwd`
//! is inside a repo (fast, respects .gitignore for free — see the module
//! docs in mod.rs for measured timing), falling back to a small manual
//! walk otherwise.

use std::path::Path;

const MAX_FILES: usize = 20_000;
const MAX_DEPTH: usize = 12;
const IGNORED_DIRS: &[&str] = &[".git", "target", "node_modules"];

/// Relative file paths under `cwd`, project-scoped.
pub fn list_project_files(cwd: &Path) -> Vec<String> {
    git_ls_files(cwd).unwrap_or_else(|| {
        let mut out = Vec::new();
        walk(cwd, cwd, &mut out, 0);
        out
    })
}

/// `git ls-files -z --cached --others --exclude-standard`, or `None` if
/// `cwd` isn't inside a git repo (or `git` isn't on PATH). `-z` (NUL
/// separated) so filenames containing spaces/newlines parse correctly.
fn git_ls_files(cwd: &Path) -> Option<Vec<String>> {
    let out = std::process::Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(cwd)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(
        out.stdout
            .split(|&b| b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect(),
    )
}

/// Small manual walk for non-git directories: skips dotfiles and a short
/// built-in ignore list. Not exhaustive or .gitignore-aware — this is the
/// uncommon fallback path, not the primary one.
fn walk(root: &Path, dir: &Path, out: &mut Vec<String>, depth: usize) {
    if depth > MAX_DEPTH || out.len() >= MAX_FILES {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if out.len() >= MAX_FILES {
            return;
        }
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || IGNORED_DIRS.contains(&name.as_ref()) {
            continue;
        }
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            walk(root, &path, out, depth + 1);
        } else if file_type.is_file()
            && let Ok(rel) = path.strip_prefix(root)
        {
            out.push(rel.to_string_lossy().into_owned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("m-files-test-{name}-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn walk_fallback_skips_dotfiles_and_ignored_dirs() {
        let dir = tmp_dir("walk1");
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::create_dir_all(dir.join("target")).unwrap();
        fs::create_dir_all(dir.join(".git")).unwrap();
        fs::write(dir.join("src/lib.rs"), "").unwrap();
        fs::write(dir.join("target/junk"), "").unwrap();
        fs::write(dir.join(".git/HEAD"), "").unwrap();
        fs::write(dir.join("README.md"), "").unwrap();

        let mut out = Vec::new();
        walk(&dir, &dir, &mut out, 0);
        out.sort();
        assert_eq!(out, vec!["README.md".to_string(), "src/lib.rs".to_string()]);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn git_ls_files_returns_none_outside_a_repo() {
        let dir = tmp_dir("notgit");
        assert_eq!(git_ls_files(&dir), None);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn git_ls_files_lists_this_repo() {
        // crates/m-tui's own manifest dir is inside the m git repo.
        let cwd = std::env::current_dir().unwrap();
        let files = git_ls_files(&cwd).expect("m-tui is checked out inside a git repo");
        assert!(!files.is_empty());
        assert!(files.iter().any(|f| f.ends_with("Cargo.toml")));
    }
}
