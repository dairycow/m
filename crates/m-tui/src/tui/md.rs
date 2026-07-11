//! Streaming-friendly markdown → styled, width-wrapped lines.
//! Own wrapping (not Paragraph::wrap) so scroll math is exact.

use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use super::{hl, theme};

pub fn render(md: &str, width: u16) -> Vec<Line<'static>> {
    let width = width.max(10) as usize;
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut cur: Vec<Span<'static>> = Vec::new();

    let mut bold = 0u32;
    let mut italic = 0u32;
    let mut heading: Option<u32> = None;
    let mut list_stack: Vec<Option<u64>> = Vec::new();
    let mut quote_depth = 0u32;
    let mut code_block: Option<(String, String)> = None; // (lang, text)
    let mut needs_blank = false;

    let style = |bold: u32, italic: u32, heading: Option<u32>| -> Style {
        let mut s = if heading.is_some() { theme::heading() } else { Style::default() };
        if bold > 0 {
            s = s.add_modifier(Modifier::BOLD);
        }
        if italic > 0 {
            s = s.add_modifier(Modifier::ITALIC);
        }
        s
    };

    let indent = |list_stack: &[Option<u64>], quote_depth: u32| -> String {
        let mut p = String::new();
        for _ in 0..quote_depth {
            p.push_str("│ ");
        }
        if !list_stack.is_empty() {
            p.push_str(&"  ".repeat(list_stack.len().saturating_sub(1)));
        }
        p
    };

    macro_rules! flush {
        () => {
            if !cur.is_empty() {
                let prefix = indent(&list_stack, quote_depth);
                out.extend(wrap_spans(std::mem::take(&mut cur), width, &prefix));
            }
        };
    }
    macro_rules! blank {
        () => {
            if needs_blank && !out.is_empty() {
                out.push(Line::default());
            }
            #[allow(unused_assignments)]
            {
                needs_blank = false;
            }
        };
    }

    let opts = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES;
    for ev in Parser::new_ext(md, opts) {
        match ev {
            Event::Start(tag) => match tag {
                Tag::Paragraph => {
                    blank!();
                }
                Tag::Heading { level, .. } => {
                    blank!();
                    heading = Some(level as u32);
                    let marks = "#".repeat(level as usize);
                    cur.push(Span::styled(format!("{marks} "), theme::heading()));
                }
                Tag::BlockQuote(_) => {
                    blank!();
                    quote_depth += 1;
                }
                Tag::CodeBlock(kind) => {
                    flush!();
                    blank!();
                    let lang = match kind {
                        CodeBlockKind::Fenced(l) => l.to_string(),
                        CodeBlockKind::Indented => String::new(),
                    };
                    code_block = Some((lang, String::new()));
                }
                Tag::List(start) => {
                    if list_stack.is_empty() {
                        blank!();
                    } else {
                        flush!();
                    }
                    list_stack.push(start);
                }
                Tag::Item => {
                    flush!();
                    let marker = match list_stack.last_mut() {
                        Some(Some(n)) => {
                            let m = format!("{n}. ");
                            *list_stack.last_mut().unwrap() = Some(*n + 1);
                            m
                        }
                        _ => "• ".to_string(),
                    };
                    cur.push(Span::styled(marker, theme::accent()));
                }
                Tag::Emphasis => italic += 1,
                Tag::Strong => bold += 1,
                Tag::Link { .. } => {}
                _ => {}
            },
            Event::End(tag) => match tag {
                TagEnd::Paragraph => {
                    flush!();
                    needs_blank = true;
                }
                TagEnd::Heading(_) => {
                    flush!();
                    heading = None;
                    needs_blank = true;
                }
                TagEnd::BlockQuote(_) => {
                    flush!();
                    quote_depth = quote_depth.saturating_sub(1);
                    needs_blank = true;
                }
                TagEnd::CodeBlock => {
                    if let Some((lang, text)) = code_block.take() {
                        out.extend(hl::highlight(&text, &lang));
                        needs_blank = true;
                    }
                }
                TagEnd::List(_) => {
                    flush!();
                    list_stack.pop();
                    if list_stack.is_empty() {
                        needs_blank = true;
                    }
                }
                TagEnd::Item => {
                    flush!();
                }
                TagEnd::Emphasis => italic = italic.saturating_sub(1),
                TagEnd::Strong => bold = bold.saturating_sub(1),
                _ => {}
            },
            Event::Text(t) => {
                if let Some((_, buf)) = &mut code_block {
                    buf.push_str(&t);
                } else {
                    cur.push(Span::styled(t.into_string(), style(bold, italic, heading)));
                }
            }
            Event::Code(t) => {
                cur.push(Span::styled(t.into_string(), theme::code_inline()));
            }
            Event::SoftBreak => {
                cur.push(Span::raw(" "));
            }
            Event::HardBreak => {
                flush!();
            }
            Event::Rule => {
                flush!();
                blank!();
                out.push(Line::styled("─".repeat(width.min(40)), theme::dim()));
                needs_blank = true;
            }
            Event::TaskListMarker(done) => {
                cur.push(Span::styled(
                    if done { "[x] " } else { "[ ] " }.to_string(),
                    theme::accent(),
                ));
            }
            _ => {}
        }
    }
    // Unterminated code fence while streaming: show what we have.
    if let Some((lang, text)) = code_block.take() {
        out.extend(hl::highlight(&text, &lang));
    }
    flush!();
    out
}

/// Wrap spans to `width`, breaking at spaces where possible. `prefix` is
/// prepended to the first line; continuation lines get equal-width padding.
pub fn wrap_spans(spans: Vec<Span<'static>>, width: usize, prefix: &str) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let cont = " ".repeat(prefix.width());
    let mut cur: Vec<Span<'static>> = Vec::new();
    let mut cur_w = if prefix.is_empty() {
        0
    } else {
        cur.push(Span::styled(prefix.to_string(), theme::dim()));
        prefix.width()
    };

    for span in spans {
        let style = span.style;
        // Split span content into words, preserving spaces.
        let mut word = String::new();
        let push_word = |word: &mut String,
                             cur: &mut Vec<Span<'static>>,
                             cur_w: &mut usize,
                             lines: &mut Vec<Line<'static>>| {
            if word.is_empty() {
                return;
            }
            let w = word.width();
            if *cur_w + w > width && *cur_w > cont.width() {
                lines.push(Line::from(std::mem::take(cur)));
                if !cont.is_empty() {
                    cur.push(Span::raw(cont.clone()));
                }
                *cur_w = cont.width();
                // Drop the leading space of a wrapped word.
                if word.starts_with(' ') {
                    let trimmed = word.trim_start().to_string();
                    *word = trimmed;
                }
            }
            if !word.is_empty() {
                *cur_w += word.width();
                cur.push(Span::styled(std::mem::take(word), style));
            }
        };
        for ch in span.content.chars() {
            if ch == ' ' {
                push_word(&mut word, &mut cur, &mut cur_w, &mut lines);
                word.push(' ');
            } else if ch == '\n' {
                push_word(&mut word, &mut cur, &mut cur_w, &mut lines);
                lines.push(Line::from(std::mem::take(&mut cur)));
                if !cont.is_empty() {
                    cur.push(Span::raw(cont.clone()));
                }
                cur_w = cont.width();
            } else {
                word.push(ch);
            }
        }
        push_word(&mut word, &mut cur, &mut cur_w, &mut lines);
    }
    if cur.iter().any(|s| !s.content.trim().is_empty()) || lines.is_empty() {
        lines.push(Line::from(cur));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(lines: &[Line<'_>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect()
    }

    #[test]
    fn wraps_long_paragraph() {
        let lines = render("one two three four five six seven eight", 12);
        let t = text_of(&lines);
        assert!(t.len() > 2, "{t:?}");
        for l in &t {
            assert!(l.width() <= 12, "line too wide: {l:?}");
        }
    }

    #[test]
    fn renders_lists_and_headings() {
        let lines = render("# Title\n\n- alpha\n- beta\n", 40);
        let t = text_of(&lines);
        assert!(t.iter().any(|l| l.contains("# Title")));
        assert!(t.iter().any(|l| l.contains("• alpha")));
    }

    #[test]
    fn code_block_survives_streaming_cutoff() {
        let lines = render("```python\nprint('hi')\n", 40);
        let t = text_of(&lines);
        assert!(t.iter().any(|l| l.contains("print")), "{t:?}");
    }
}
