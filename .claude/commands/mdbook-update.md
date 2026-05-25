Update the mdbook knowledge graph at `D:\tensorrt\book\`.

## Instructions

1. **Read** `D:\tensorrt\book\src\SUMMARY.md` to understand the current structure.

2. **Interpret the user's request**: `$ARGUMENTS`
   - If the argument names an existing page (e.g. "preprocessing", "results"), update that page.
   - If the argument describes new content (e.g. "add a page about INT8 calibration"), create the new `.md` file and add it to `SUMMARY.md`.
   - If no argument is given, ask the user what they want to update.

3. **Edit or create** the relevant markdown file(s) under `book/src/`. Use Mermaid diagrams where they help explain concepts — wrap them in ` ```mermaid ` blocks.

4. If adding a new page, insert it into `SUMMARY.md` under the most relevant section.

5. **Rebuild** the book by running:
   ```
   cd D:\tensorrt\book && mdbook build
   ```
   Report success or any errors.

## Conventions

- All prose in Traditional Chinese (繁體中文) unless the user writes in English.
- Diagrams: prefer `flowchart`, `sequenceDiagram`, or `graph` depending on what best represents the concept.
- No inline HTML — keep everything in standard Markdown + Mermaid.
- Keep pages focused: one concept per page, cross-link with relative paths when referencing other pages.

## Mermaid v11 Rules (strictly enforced)

- **Subgraph labels** containing spaces, Chinese characters, parentheses, or commas **must be quoted**: `subgraph "My Label（說明）"` — bare labels cause "Syntax error in text".
- **Node label line breaks**: use `<br/>` inside bracket syntax, never `\n`. Example: `A["line one<br/>line two"]`.
- **Node labels with special chars** (colons, parentheses, `+`, `×`): wrap in double quotes inside the brackets: `A["Conv+BN+ReLU<br/>單一 Kernel"]`.
- Test mentally: if a label has any character outside `[A-Za-z0-9_]`, add quotes.
