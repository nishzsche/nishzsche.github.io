# To do

### Launch: "Can AI know when it's wrong?"

- [x] Analytics: GoatCounter is on (`goatcounter_code: nishzsche` in `_config.yml`; stats at nishzsche.goatcounter.com)
- [ ] Comments: enable Discussions in the repo settings, install the giscus app, fill in `giscus.repo_id` and `giscus.category_id` in `_config.yml`
- [ ] Move `review/can-ai-know-when-its-wrong.md` to `_posts/YYYY-MM-DD-can-ai-know-when-its-wrong.md`
  - drop `permalink`, `noindex` and `hypothesis` from the front matter, and set `date`
  - remove the "Draft for review" note at the top
  - keep `description`, `image` and `image_alt` (they drive the link preview), and `toc: true` (the contents list)
- [ ] Check the preview after deploy: LinkedIn Post Inspector and opengraph.xyz
- [ ] Update the About page background

### Site

- [ ] Notebook pipeline: any push to `dev` that touches `notebooks/*.ipynb` re-converts every notebook into a new, today-dated post in `_posts/`, then merges `dev` into `gh-pages`. Decide whether the two old notebooks should be published before the first notebook change, or move them out of `notebooks/`
- [ ] Remove `_includes/sidebar.html` (left over from Lanyon, not used by any layout)

### Done ✓

- [x] Link previews: per-page description and 1200x630 image, large X card
- [x] Clean pip install cells (in the notebook workflow)
- [x] Remove Google Colab cells (in the notebook workflow)
- [x] Extract the title from the file name (in the notebook workflow)
- [x] Change index page format to publish blogs one by one
