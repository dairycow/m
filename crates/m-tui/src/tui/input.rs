//! Multi-line input editor: emacs-ish bindings, history, kill ring.

use unicode_width::UnicodeWidthStr;

#[derive(Default)]
pub struct Editor {
    text: String,
    /// Byte offset of the cursor within `text`.
    cursor: usize,
    kill: String,
    history: Vec<String>,
    /// None = editing a fresh entry; Some(i) = browsing history[i].
    hist_idx: Option<usize>,
    stash: String,
}

impl Editor {
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn set(&mut self, s: &str) {
        self.text = s.to_string();
        self.cursor = self.text.len();
    }

    pub fn take(&mut self) -> String {
        let t = std::mem::take(&mut self.text);
        self.cursor = 0;
        self.hist_idx = None;
        if !t.trim().is_empty() {
            self.history.push(t.clone());
        }
        t
    }

    pub fn insert(&mut self, ch: char) {
        self.text.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    pub fn insert_str(&mut self, s: &str) {
        self.text.insert_str(self.cursor, s);
        self.cursor += s.len();
    }

    fn prev_char(&self) -> Option<char> {
        self.text[..self.cursor].chars().next_back()
    }
    fn next_char(&self) -> Option<char> {
        self.text[self.cursor..].chars().next()
    }

    pub fn backspace(&mut self) {
        if let Some(c) = self.prev_char() {
            self.cursor -= c.len_utf8();
            self.text.remove(self.cursor);
        }
    }

    pub fn delete(&mut self) {
        if self.next_char().is_some() {
            self.text.remove(self.cursor);
        }
    }

    pub fn left(&mut self) {
        if let Some(c) = self.prev_char() {
            self.cursor -= c.len_utf8();
        }
    }

    pub fn right(&mut self) {
        if let Some(c) = self.next_char() {
            self.cursor += c.len_utf8();
        }
    }

    pub fn home(&mut self) {
        self.cursor = self.text[..self.cursor].rfind('\n').map(|i| i + 1).unwrap_or(0);
    }

    pub fn end(&mut self) {
        self.cursor =
            self.text[self.cursor..].find('\n').map(|i| self.cursor + i).unwrap_or(self.text.len());
    }

    pub fn word_left(&mut self) {
        let before = &self.text[..self.cursor];
        let trimmed = before.trim_end_matches(|c: char| !c.is_alphanumeric());
        let base = trimmed.trim_end_matches(|c: char| c.is_alphanumeric());
        self.cursor = base.len().min(self.cursor.saturating_sub(1)).max(base.len());
    }

    pub fn word_right(&mut self) {
        let after = &self.text[self.cursor..];
        let skip_non = after.len() - after.trim_start_matches(|c: char| !c.is_alphanumeric()).len();
        let rest = &after[skip_non..];
        let skip_word = rest.len() - rest.trim_start_matches(|c: char| c.is_alphanumeric()).len();
        self.cursor += skip_non + skip_word;
    }

    pub fn delete_word_back(&mut self) {
        let end = self.cursor;
        self.word_left();
        self.kill = self.text[self.cursor..end].to_string();
        self.text.replace_range(self.cursor..end, "");
    }

    pub fn kill_to_start(&mut self) {
        let start = self.text[..self.cursor].rfind('\n').map(|i| i + 1).unwrap_or(0);
        self.kill = self.text[start..self.cursor].to_string();
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    pub fn kill_to_end(&mut self) {
        let end =
            self.text[self.cursor..].find('\n').map(|i| self.cursor + i).unwrap_or(self.text.len());
        self.kill = self.text[self.cursor..end].to_string();
        self.text.replace_range(self.cursor..end, "");
    }

    pub fn yank(&mut self) {
        let k = self.kill.clone();
        self.insert_str(&k);
    }

    /// (row, col-in-chars) of the cursor.
    pub fn cursor_rc(&self) -> (usize, usize) {
        let before = &self.text[..self.cursor];
        let row = before.matches('\n').count();
        let col_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
        (row, before[col_start..].width())
    }

    pub fn lines(&self) -> Vec<&str> {
        if self.text.is_empty() { vec![""] } else { self.text.split('\n').collect() }
    }

    /// Move up a line, or into history when already at the top line.
    pub fn up(&mut self) -> bool {
        let (row, col) = self.cursor_rc();
        if row > 0 {
            self.move_to_rc(row - 1, col);
            return true;
        }
        // history
        let next = match self.hist_idx {
            None if !self.history.is_empty() => {
                self.stash = self.text.clone();
                Some(self.history.len() - 1)
            }
            Some(i) if i > 0 => Some(i - 1),
            other => other,
        };
        if let Some(i) = next {
            self.hist_idx = Some(i);
            self.set(&self.history[i].clone());
            return true;
        }
        false
    }

    pub fn down(&mut self) -> bool {
        let (row, col) = self.cursor_rc();
        if row + 1 < self.lines().len() {
            self.move_to_rc(row + 1, col);
            return true;
        }
        match self.hist_idx {
            Some(i) if i + 1 < self.history.len() => {
                self.hist_idx = Some(i + 1);
                self.set(&self.history[i + 1].clone());
                true
            }
            Some(_) => {
                self.hist_idx = None;
                let stash = std::mem::take(&mut self.stash);
                self.set(&stash);
                true
            }
            None => false,
        }
    }

    fn move_to_rc(&mut self, row: usize, col: usize) {
        let mut offset = 0usize;
        for (i, line) in self.text.split('\n').enumerate() {
            if i == row {
                let byte: usize =
                    line.chars().take(col).map(char::len_utf8).sum();
                self.cursor = offset + byte;
                return;
            }
            offset += line.len() + 1;
        }
        self.cursor = self.text.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_and_history() {
        let mut e = Editor::default();
        e.insert_str("hello world");
        e.word_left();
        assert_eq!(&e.text()[..e.cursor], "hello ");
        e.kill_to_end();
        assert_eq!(e.text(), "hello ");
        e.yank();
        assert_eq!(e.text(), "hello world");
        let t = e.take();
        assert_eq!(t, "hello world");
        assert!(e.is_empty());
        assert!(e.up()); // recalls from history
        assert_eq!(e.text(), "hello world");
    }

    #[test]
    fn multiline_cursor() {
        let mut e = Editor::default();
        e.insert_str("ab\ncdef");
        assert_eq!(e.cursor_rc(), (1, 4));
        e.home();
        assert_eq!(e.cursor_rc(), (1, 0));
        assert!(e.up());
        assert_eq!(e.cursor_rc(), (0, 0));
        e.end();
        assert_eq!(e.cursor_rc(), (0, 2));
    }
}
