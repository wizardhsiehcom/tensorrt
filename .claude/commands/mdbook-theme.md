Change the mdbook theme for the knowledge graph at `D:\tensorrt\book\`.

## Available Themes

| Name   | Description                        |
|--------|------------------------------------|
| `light`  | Default light theme (mdBook stock) |
| `rust`   | Warm rust/orange tones             |
| `coal`   | Dark grey, easy on the eyes        |
| `navy`   | Dark blue                          |
| `ayu`    | Dark with amber highlights         |

## Instructions

1. **Read** `D:\tensorrt\book\book.toml` to check the current theme settings under `[output.html]`.

2. **Interpret the argument**: `$ARGUMENTS`
   - If a valid theme name is given (light/rust/coal/navy/ayu), apply it.
   - If no argument or an invalid name is given, list the available themes and show the current one, then ask the user which they want.

3. **Update `book.toml`**: Set or update these two keys under `[output.html]`:
   - `default-theme` — the theme used on first load
   - `preferred-dark-theme` — keep in sync: use `coal` for light/rust, keep the chosen dark theme otherwise

   Example result for `navy`:
   ```toml
   [output.html]
   default-theme = "navy"
   preferred-dark-theme = "navy"
   ```

4. **Rebuild** the book:
   ```
   cd D:\tensorrt\book && mdbook build
   ```
   Report the new theme and any build errors.

## Rules

- Only modify `default-theme` and `preferred-dark-theme` — leave all other keys in `book.toml` untouched.
- Never add duplicate keys; if the key already exists, edit it in place.
- Report: old theme → new theme, and confirm the build succeeded.
