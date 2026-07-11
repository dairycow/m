//! One coherent dark theme. Colors chosen to read on any dark terminal.

use ratatui::style::{Color, Modifier, Style};

pub const ACCENT: Color = Color::Cyan;
pub const DIM: Color = Color::DarkGray;
pub const USER: Color = Color::Green;
pub const TOOL: Color = Color::Yellow;
pub const ERROR: Color = Color::Red;
pub const ADD: Color = Color::Green;
pub const DEL: Color = Color::Red;

pub fn dim() -> Style {
    Style::default().fg(DIM)
}

pub fn thinking() -> Style {
    Style::default().fg(DIM).add_modifier(Modifier::ITALIC)
}

pub fn user_tag() -> Style {
    Style::default().fg(USER).add_modifier(Modifier::BOLD)
}

pub fn tool_tag() -> Style {
    Style::default().fg(TOOL)
}

pub fn error() -> Style {
    Style::default().fg(ERROR)
}

pub fn accent() -> Style {
    Style::default().fg(ACCENT)
}

pub fn heading() -> Style {
    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
}

pub fn code_inline() -> Style {
    Style::default().fg(Color::LightYellow)
}
