---
name: mermaid
description: Render Mermaid diagrams (flowchart, sequence, class, state, ER, …) to SVG/PNG in milliseconds with the mmdr CLI. Use for architecture sketches and planning whenever a diagram communicates better than prose.
---

# Mermaid diagrams with mmdr

`mmdr` is a pure-Rust Mermaid renderer (no browser, ~1–5 ms per diagram).
Install once with `cargo install mermaid-rs-renderer`.

Workflow:

1. Write the diagram source to a file, e.g. `docs/arch.mmd`.
2. Render it: `mmdr -i docs/arch.mmd -o docs/arch.svg`
   (raster: `-e png -o docs/arch.png`, wider: `-w 1600`).
3. Give the user the output path (they open it with `xdg-open <file>`),
   and keep the `.mmd` source next to it so the diagram stays editable.

Notes:

- Themes: `-t dark|forest|neutral|modern` (default is light).
- A non-zero exit means a parse error, reported on stderr with the
  offending line — fix the `.mmd` and re-render rather than giving up.
- For planning discussions, iterate: render, show the path, adjust the
  `.mmd` from feedback, re-render to the same output path.
