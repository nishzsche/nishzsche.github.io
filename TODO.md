# To do

### Launch: "Can AI know when it's wrong?"

- [x] Analytics: GoatCounter is on (`goatcounter_code: nishzsche` in `_config.yml`; stats at nishzsche.goatcounter.com)
- [x] Comments: Giscus is on, stored as Discussions in this repo, nishzsche/nishzsche.github.io (Announcements). It shows on published posts, not on review copies
- [x] Published as `_posts/2026-11-23-can-ai-know-when-its-wrong.md`, the review URL redirects; the deep dive lives in the project repo (writeup/deep_dive.md)
  - drop `permalink`, `noindex` and `hypothesis` from the front matter, and set `date`
  - remove the "Draft for review" note at the top
  - keep `description`, `image` and `image_alt` (they drive the link preview), and `toc: true` (the contents list)
- [ ] Check the preview after deploy: LinkedIn Post Inspector and opengraph.xyz
- [x] About page: credit-risk background and the experiment linked

### Site

- [x] Giscus app installed on nishzsche.github.io
- [x] Notebook pipeline removed (it republished old notebooks as new posts); `notebooks/` stays, unpublished
- [x] Removed Lanyon leftovers: `sidebar.html`, `lanyon.css`, `poole.css`, `script.js`
- [x] Repo docs and `notebooks/` excluded from the site; sitemap.xml and robots.txt on; MathJax only with `math: true`

### Done ✓

- [x] Link previews: per-page description and 1200x630 image, large X card
- [x] Clean pip install cells (in the notebook workflow)
- [x] Remove Google Colab cells (in the notebook workflow)
- [x] Extract the title from the file name (in the notebook workflow)
- [x] Change index page format to publish blogs one by one
