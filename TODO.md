# To do

### Launch: "Can AI know when it's wrong?"

- [x] Analytics: GoatCounter is on (`goatcounter_code: nishzsche` in `_config.yml`; stats at nishzsche.goatcounter.com)
- [x] Comments: Giscus is on, stored as Discussions in nishzsche/can-ai-know-when-its-wrong (Announcements). It shows on published posts, not on review copies
- [x] Published as `_posts/2026-11-23-can-ai-know-when-its-wrong.md`, with a deep dive alongside; the review URL redirects
  - drop `permalink`, `noindex` and `hypothesis` from the front matter, and set `date`
  - remove the "Draft for review" note at the top
  - keep `description`, `image` and `image_alt` (they drive the link preview), and `toc: true` (the contents list)
- [ ] Check the preview after deploy: LinkedIn Post Inspector and opengraph.xyz
- [x] About page: credit-risk background and the experiment linked

### Site

- [x] Notebook pipeline removed (it republished old notebooks as new posts); `notebooks/` stays, unpublished
- [ ] Remove `_includes/sidebar.html` (left over from Lanyon, not used by any layout)

### Done ✓

- [x] Link previews: per-page description and 1200x630 image, large X card
- [x] Clean pip install cells (in the notebook workflow)
- [x] Remove Google Colab cells (in the notebook workflow)
- [x] Extract the title from the file name (in the notebook workflow)
- [x] Change index page format to publish blogs one by one
