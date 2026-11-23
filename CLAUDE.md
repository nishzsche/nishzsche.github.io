# Project Instructions

Guidelines for any AI assistant working in this repository.

## Writing conventions

- **No em-dashes.** Never use the `—` character anywhere in the codebase.
  Use a hyphen with spaces (` - `), a comma, or a colon depending on context.
  This applies to all file types: Markdown, HTML, CSS comments, YAML, scripts.

- Keep prose direct and concise. No filler phrases.

## Repository conventions

- **Working branch is `dev`.** All changes go to `dev`.
  Never commit directly to `main` or `gh-pages` - those are deployed by GitHub Pages.

- The working directory for this project is
  `C:/Users/nisha/Documents/Education & Learning/GitHub/nishzsche.github.io`.

## Site structure

- Jekyll static site, deployed via GitHub Pages.
- Layouts: `_layouts/` · Styles: `public/css/main.css` · Published posts: `_posts/`
- Short posts originate from GitHub Issues. Longer experiment posts are written in their own
  project repo and copied to `_posts/` by hand. There is no notebook pipeline any more:
  `notebooks/` holds two old notebooks and is not published.
- See `WRITING.md` for the full content and publishing workflow.
