//! Plain subsequence fuzzy matching for the `@` file picker — the standard
//! file-picker UX (fzf, vim ctrlp, VS Code quick-open). Deliberately no
//! typo/substitution tolerance; that's a fuzzy spellchecker, not what a
//! file picker needs.

const MATCH: i64 = 16;
const CONSECUTIVE: i64 = 12;
const BOUNDARY: i64 = 10;
const PREFIX: i64 = 20;

/// Score `candidate` against `query`, or `None` if `query`'s characters
/// don't all appear in `candidate`, in order (ASCII-lowercase comparison
/// only, to avoid Unicode-casing length mismatches like `ß` -> `ss`).
/// Greedy left-to-right match — for each query character, take the
/// earliest remaining match in `candidate` — not a full DP alignment;
/// good enough for filtering a file list, not trying to be optimal.
/// Higher is better; an empty query matches everything with score 0.
pub fn score(candidate: &str, query: &str) -> Option<i64> {
    if query.is_empty() {
        return Some(0);
    }
    let hay: Vec<u8> = candidate.bytes().map(|b| b.to_ascii_lowercase()).collect();
    let needle: Vec<u8> = query.bytes().map(|b| b.to_ascii_lowercase()).collect();

    let mut total = 0i64;
    let mut hay_idx = 0usize;
    let mut prev_match: Option<usize> = None;
    let mut first_match: Option<usize> = None;

    for &nc in &needle {
        let found = hay[hay_idx..].iter().position(|&hc| hc == nc)?;
        let pos = hay_idx + found;
        total += MATCH;
        if prev_match == pos.checked_sub(1) {
            total += CONSECUTIVE;
        }
        if pos == 0 || matches!(hay[pos - 1], b'/' | b'-' | b'_' | b'.' | b' ') {
            total += BOUNDARY;
        }
        first_match.get_or_insert(pos);
        prev_match = Some(pos);
        hay_idx = pos + 1;
    }

    if first_match == Some(0) {
        total += PREFIX;
    }
    // Slight bias toward shorter candidates: "agent.rs" should outrank
    // "agent_helpers.rs" for the same query when both match equally well.
    total -= hay.len() as i64 / 4;
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_and_prefix_match() {
        assert!(score("agent.rs", "agent").is_some());
        assert!(score("agent.rs", "agent").unwrap() > score("xagentx.rs", "agent").unwrap());
    }

    #[test]
    fn matches_across_path_boundaries() {
        // The whole point: "@core/agent" should hit a nested file.
        assert!(score("crates/m-core/src/agent.rs", "core/agent").is_some());
    }

    #[test]
    fn rejects_out_of_order_characters() {
        assert_eq!(score("agent.rs", "gaent"), None);
    }

    #[test]
    fn rejects_missing_characters() {
        assert_eq!(score("foo.rs", "xyz"), None);
    }

    #[test]
    fn case_insensitive() {
        assert!(score("Agent.rs", "agent").is_some());
    }

    #[test]
    fn empty_query_matches_everything_with_zero_score() {
        assert_eq!(score("anything.rs", ""), Some(0));
    }

    #[test]
    fn boundary_and_consecutive_matches_outrank_scattered_ones() {
        // "mod" as a contiguous, '/'-boundary-aligned run should score
        // higher than the same three letters scattered with no boundary
        // or consecutive-run bonuses anywhere in the match.
        let boundary = score("tui/mod.rs", "mod").unwrap();
        let scattered = score("aaamaaaoaaadaaa", "mod").unwrap();
        assert!(boundary > scattered);
    }
}
