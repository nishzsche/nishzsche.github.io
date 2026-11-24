# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""My drawings on the blog: one per post, picked at random once, and the About sketchbook.

    uv run scripts/drawings.py              # process new drawings, then give drawing-less posts one
    uv run scripts/drawings.py --redo ID    # re-crop one drawing (after changing CROPS below)
    uv run scripts/drawings.py --redo all   # re-make everything (after changing the sizes)

1. Originals go in _drawings/ (git-ignored: phone photos are large and carry location data).
   They're my Inktober 2026 pages, plus home.jpg: the park sketch held up in the park it shows.
2. Each new drawing is cropped to the sketchbook page and squared, then saved as
   public/drawings/<id>.jpg (1024 px: behind the post title, and the sketchbook's large view) and
   public/drawings/thumbs/<id>.jpg (400 px: the sketchbook's thumbnail strip), with all metadata
   (including location) stripped. The page is found as the bright, low-saturation area against
   the dark cutting mat; CROPS overrides that where it goes wrong. home.jpg is kept whole (the
   point is the sketch beside the real scene): 1024 px tall, with a square thumbnail of its lower
   part, where the sketch is.
3. Every post in _posts/ without `icon:` in its front matter gets a random drawing that no other
   post uses (once all are used, one of the least used), written into the front matter. It stays
   from then on; delete the line to draw again. home.jpg is never picked.

The sketchbook shows every thumbnail, home last; captions come from _data/drawings.yml.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "_drawings"
OUT = ROOT / "public" / "drawings"
POSTS = ROOT / "_posts"
THUMBS = OUT / "thumbs"
DRAWING_PX = 1024
THUMB_PX = 400
HOME = "home"  # the park photo: in the sketchbook, never a post's drawing
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Manual crops for photos the page finder gets wrong: id -> (left, top, right, bottom) as fractions
# of the photo. The result is still squared around its centre.
CROPS: dict[str, tuple[float, float, float, float]] = {
    "20261004-153005": (0.15, 0.27, 0.66, 0.71),  # cacti: page photographed at an angle
    "20261028-175141": (0.22, 0.31, 0.74, 0.70),  # watercolour animal: narrow, tilted page
    "20261030-163259": (0.255, 0.165, 0.68, 0.48),  # two portraits on one page: the top one
    "20261027-155301": (0.29, 0.09, 0.79, 0.76),  # trench coat: open book, mat along the top
}


def drawing_id(path: Path) -> str:
    """Stable name: the date and time from a Pixel file name (PXL_20261001_125135254 ->
    20261001-125135), otherwise the file name, lower-cased with runs of other characters as '-'."""
    m = re.search(r"PXL_(\d{8})_(\d{6})", path.stem)
    if m:
        return f"{m[1]}-{m[2]}"
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")


def load(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def clean(im: Image.Image) -> Image.Image:
    """A copy built from pixels only, so no EXIF (location, camera) can reach the saved file."""
    out = Image.new("RGB", im.size)
    out.paste(im)
    return out


def page_box(im: Image.Image) -> tuple[int, int, int, int]:
    """Bounding box of the sketchbook page: bright, low-saturation pixels, after an erosion that
    removes the cutting mat's thin grid lines."""
    small = im.copy()
    small.thumbnail((600, 600))
    _, s, v = small.convert("HSV").split()
    paper = ImageChops.multiply(v.point(lambda x: 255 if x > 150 else 0),
                                s.point(lambda x: 255 if x < 80 else 0))  # fmt: skip
    paper = paper.filter(ImageFilter.MinFilter(9)).filter(ImageFilter.MaxFilter(9))
    box = paper.getbbox() or (0, 0, *small.size)
    k = im.width / small.width
    return tuple(round(c * k) for c in box)  # type: ignore[return-value]


def square_crop(im: Image.Image, did: str) -> Image.Image:
    if did in CROPS:
        l, t, r, b = CROPS[did]
        box = (l * im.width, t * im.height, r * im.width, b * im.height)
    else:
        l, t, r, b = page_box(im)
        inset = 0.07 * min(r - l, b - t)  # drop the page edge and its rounded corners
        box = (l + inset, t + inset, r - inset, b - inset)
    l, t, r, b = box
    side = min(r - l, b - t)
    cx, cy = (l + r) / 2, (t + b) / 2
    return im.crop((round(cx - side / 2), round(cy - side / 2),
                    round(cx + side / 2), round(cy + side / 2)))  # fmt: skip


def save(im: Image.Image, path: Path, quality: int = 80) -> None:
    clean(im).save(path, quality=quality, optimize=True, progressive=True)


def process(redo: set[str]) -> list[str]:
    """Make the sizes for drawings that don't have them yet (or are in `redo`)."""
    THUMBS.mkdir(parents=True, exist_ok=True)
    every = "all" in redo
    made = []
    for path in sorted(p for p in SRC.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        did = HOME if path.stem.lower() == HOME else drawing_id(path)
        out, thumb = OUT / f"{did}.jpg", THUMBS / f"{did}.jpg"
        if out.exists() and thumb.exists() and did not in redo and not every:
            continue
        im = load(path)
        if did == HOME:
            whole = im.copy()
            whole.thumbnail((DRAWING_PX, DRAWING_PX))
            save(whole, out)
            side = min(im.size)  # the square at the bottom of the photo, where the sketch is
            crop = im.crop(((im.width - side) // 2, im.height - side,
                            (im.width + side) // 2, im.height))  # fmt: skip
            save(crop.resize((THUMB_PX, THUMB_PX), Image.LANCZOS), thumb)
        else:
            crop = square_crop(im, did)
            save(crop.resize((DRAWING_PX, DRAWING_PX), Image.LANCZOS), out)
            save(crop.resize((THUMB_PX, THUMB_PX), Image.LANCZOS), thumb)
        made.append(did)
    return made


FRONT = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.S)


def assign() -> list[tuple[str, str]]:
    """Give each post without an icon a random drawing, least used first."""
    icons = sorted(f"/public/drawings/{p.name}" for p in OUT.glob("*.jpg") if p.stem != HOME)
    if not icons:
        return []
    posts = sorted(POSTS.glob("*.md"))
    used = Counter()
    todo = []
    for post in posts:
        m = FRONT.match(post.read_text(encoding="utf-8"))
        if not m:
            continue
        icon = re.search(r"^icon:\s*(\S+)", m[1], re.M)
        if icon:
            used[icon[1]] += 1
        else:
            todo.append(post)
    given = []
    for post in todo:
        fewest = min(used[i] for i in icons)
        pick = random.choice([i for i in icons if used[i] == fewest])
        used[pick] += 1
        text = post.read_text(encoding="utf-8")
        nl = "\r\n" if "\r\n" in text[:200] else "\n"
        m = FRONT.match(text)
        head_end = m.end(1)
        post.write_text(text[:head_end] + f"{nl}icon: {pick}" + text[head_end:], encoding="utf-8",
                        newline="")  # fmt: skip
        given.append((post.name, pick))
    return given


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--redo", nargs="*", default=[], help="drawing ids (including 'home') or 'all' to process again")
    args = ap.parse_args()
    for did in process(set(args.redo)):
        print(f"processed  {did}")
    for name, icon in assign():
        print(f"assigned   {icon} -> {name}")


if __name__ == "__main__":
    main()
