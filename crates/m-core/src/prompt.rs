//! The system prompt. Kept deliberately small (pi's lesson: models already
//! know the coding-agent pattern; every extra token is paid on every turn
//! and pushes against the local server's KV prefix cache).

use std::path::Path;

pub fn system_prompt(cwd: &Path, project_context: &str, skills_index: &str) -> String {
    let date = today();
    let mut p = format!(
        "You are m, a fast, minimal coding agent in a terminal.\n\
         Working directory: {cwd} (OS: linux). Today: {date}.\n\
         \n\
         Work autonomously: investigate with tools, make the change, verify it, then reply \
         with a brief summary and no further tool calls. Keep replies short; this is a terminal.\n\
         \n\
         Rules:\n\
         - Read a file before editing it. edit requires old_string to match exactly and uniquely.\n\
         - Use bash for search (grep/rg, find, ls), git, tests, and running code. \
         Non-interactive commands only.\n\
         - Never invent file contents or paths — check first.\n\
         - Prefer small, surgical edits over rewrites.\n",
        cwd = cwd.display(),
    );
    if !skills_index.is_empty() {
        p.push_str("\nSkills (load one with the skill tool when relevant):\n");
        p.push_str(skills_index);
    }
    if !project_context.is_empty() {
        p.push_str("\nProject notes:\n");
        p.push_str(project_context);
    }
    p
}

/// YYYY-MM-DD from the civil-from-days algorithm; avoids a chrono dep.
fn today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let z = secs.div_euclid(86400) + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod tests {
    #[test]
    fn date_shape() {
        let d = super::today();
        assert_eq!(d.len(), 10);
        assert!(d.starts_with("20"));
    }
}
