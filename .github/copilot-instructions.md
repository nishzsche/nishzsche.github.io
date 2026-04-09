# Copilot Instructions

Guidelines for GitHub Copilot and any AI assistant working in this repository.

## Writing conventions

- **No em-dashes.** Never use the `—` character anywhere in the codebase.
  Use a hyphen with spaces (` - `), a comma, or a colon depending on context.
  This applies to all file types: Markdown, HTML, CSS comments, YAML, scripts.

- Keep prose direct and concise. No filler phrases.

## Repository conventions

- **Working branch is `dev`.** All changes go to `dev`.
  Never commit directly to `main` or `gh-pages`.

## Site structure

- Jekyll static site deployed via GitHub Pages.
- Layouts in `_layouts/`, styles in `public/css/main.css`, posts in `_posts/`.
- Short posts come from GitHub Issues. Notebook posts use the automation pipeline
  in `.github/workflows/convert_notebook_to_markdown.yml`.
- See `WRITING.md` for the full content and publishing workflow.
