//! Lazy syntect highlighting. The syntax/theme dumps load on a background
//! thread at startup so the first frame never waits; until ready, code
//! renders unstyled.

use std::sync::OnceLock;

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::SyntaxSet;

struct Assets {
    syntaxes: SyntaxSet,
    theme: Theme,
}

static ASSETS: OnceLock<Assets> = OnceLock::new();

fn assets() -> &'static Assets {
    ASSETS.get_or_init(|| {
        let syntaxes = SyntaxSet::load_defaults_newlines();
        let mut themes = ThemeSet::load_defaults();
        let theme = themes.themes.remove("base16-eighties.dark").unwrap_or_default();
        Assets { syntaxes, theme }
    })
}

/// Call early from a background thread to warm the lazy assets.
pub fn preload() {
    std::thread::spawn(|| {
        let _ = assets();
    });
}

fn ready() -> bool {
    ASSETS.get().is_some()
}

/// Highlight a fenced code block into styled lines (2-space indented).
pub fn highlight(code: &str, lang: &str) -> Vec<Line<'static>> {
    if !ready() && lang.is_empty() {
        return plain(code);
    }
    let a = assets();
    let syntax = a
        .syntaxes
        .find_syntax_by_token(lang)
        .or_else(|| a.syntaxes.find_syntax_by_first_line(code.lines().next().unwrap_or("")))
        .unwrap_or_else(|| a.syntaxes.find_syntax_plain_text());
    let mut h = HighlightLines::new(syntax, &a.theme);
    let mut out = Vec::new();
    for line in code.lines() {
        let mut spans = vec![Span::raw("  ")];
        match h.highlight_line(line, &a.syntaxes) {
            Ok(ranges) => {
                for (style, text) in ranges {
                    let fg = style.foreground;
                    spans.push(Span::styled(
                        text.to_string(),
                        Style::default().fg(Color::Rgb(fg.r, fg.g, fg.b)),
                    ));
                }
            }
            Err(_) => spans.push(Span::raw(line.to_string())),
        }
        out.push(Line::from(spans));
    }
    out
}

fn plain(code: &str) -> Vec<Line<'static>> {
    code.lines().map(|l| Line::from(format!("  {l}"))).collect()
}
