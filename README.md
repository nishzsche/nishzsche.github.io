# Learning Curve

> *An authentic pursuit to rid the imposter in me.*

A personal ML learning blog by [Nishanth Rajamani](https://nishzsche.github.io/about).

Short posts about things that confused me and how I eventually resolved them. Occasionally longer posts walking through experiments and implementations. The approach is Feynman-ish: if I can't explain it clearly, I probably don't understand it yet.

**Live site:** [nishzsche.github.io](https://nishzsche.github.io)

---

## Technical setup

- **Generator:** [Jekyll](https://jekyllrb.com) via [GitHub Pages](https://pages.github.com)
- **Branch structure:** `dev` → working branch (all changes go here) · `main`/`gh-pages` → deployed by GitHub Pages
- **Theme:** Custom - no third-party theme dependency. Layouts in `_layouts/`, styles in `public/css/main.css`

## Running locally

```bash
# From the repo root (dev branch)
bundle install
bundle exec jekyll serve --drafts
# → http://localhost:4000
```

## Publishing workflow

### Short insight posts (phone-first)

1. Feel a confusion → open a **GitHub Issue** with label `confusion`
2. Fill in the three-move structure in the issue body (see `WRITING.md`)
3. Iterate in the issue; change label to `drafting`, then `ready`
4. When ready: copy to `_posts/YYYY-MM-DD-slug.md` → push to `dev` → close issue

### Longer experiment posts

1. Run the experiment in its own project repo (code, data, results and the write-up live there)
2. Optionally publish an unlisted review copy in `review/` (`noindex: true`) for feedback
3. Copy the finished post to `_posts/YYYY-MM-DD-slug.md`, with its figures in `public/images/<slug>/`
4. Set `description`, `image` and `image_alt` for link previews, and `toc: true` for a contents list

There's no notebook pipeline any more: `notebooks/` keeps two old notebooks for reference and isn't published.

### Writing guide

See [`WRITING.md`](WRITING.md) for:
- The three-move post structure with annotated examples
- The level test ("write to yourself 6 months ago")
- OMSCS-specific guidance
- Tag taxonomy
- Full draft workflow

---

## Repo structure

```
├── _layouts/          # default, post, page layouts
├── _includes/         # head.html (meta, OG tags, CSS)
├── _posts/            # published posts
├── _drafts/           # drafts and templates; short posts live in GitHub Issues
│   ├── _template.md   # three-move post template
│   └── INBOX.md       # local confusion capture
├── notebooks/         # old Jupyter notebooks, kept for reference (not published)
├── review/            # unlisted review copies (noindex)
├── public/css/        # main.css (custom), syntax.css (Rouge highlighting)
├── WRITING.md         # writing guide and draft workflow
└── _config.yml        # Jekyll configuration
```

---

MIT License
