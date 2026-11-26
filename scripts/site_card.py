# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""The site's default link-preview card: public/site-card.png (1200x630), og_image in _config.yml.

    uv run scripts/site_card.py

Used for pages without their own `image:` (home, About, 404); posts have their own cards. The
tagline and the byline on the paper colour, with the park sketch pasted on the right like the
drawing cards on the site. Fonts are the system's own: Segoe UI on Windows, else DejaVu Sans.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "site-card.png"
SKETCH = ROOT / "public" / "drawings" / "home.jpg"
W, H = 1200, 630
PAPER, CARD = "#faf8f3", "#fffdf9"
INK, MUTED, ACCENT = "#1c1917", "#6b645c", "#c8334a"

FONTS = {
    "bold": ["C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "regular": ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in FONTS[kind]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise SystemExit(f"no {kind} font found: add one to FONTS")


def pasted(sketch: Image.Image, height: int) -> Image.Image:
    """The sketch on a paper card with a soft shadow, turned slightly, like the site's cards."""
    sketch = sketch.convert("RGB")
    sketch.thumbnail((height, height - 24))
    pad = 10
    card = Image.new("RGB", (sketch.width + 2 * pad, sketch.height + 2 * pad), CARD)
    card.paste(sketch, (pad, pad))
    margin = 40
    layer = Image.new("RGBA", (card.width + 2 * margin, card.height + 2 * margin), (0, 0, 0, 0))
    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rectangle(
        (margin + 4, margin + 12, margin + card.width - 4, margin + card.height + 8),
        fill=(28, 25, 23, 90),
    )
    layer = Image.alpha_composite(layer, shadow.filter(ImageFilter.GaussianBlur(14)))
    layer.paste(card, (margin, margin))
    return layer.rotate(2, resample=Image.BICUBIC, expand=True)


def main() -> None:
    img = Image.new("RGB", (W, H), PAPER)
    card = pasted(Image.open(SKETCH), 560)
    img.paste(card, (W - card.width + 10, (H - card.height) // 2), card)

    d = ImageDraw.Draw(img)
    x = 72
    d.text((x, 92), "LEARNING CURVE", font=font("bold", 22), fill=MUTED)
    d.rectangle((x, 132, x + 44, 136), fill=ACCENT)
    title = font("bold", 64)
    for i, line in enumerate(("Testing what", "I think I know")):
        d.text((x, 176 + i * 80), line, font=title, fill=INK)
    sub = font("regular", 28)
    d.text((x, 370), "Experiments on AI models, with real", font=sub, fill=MUTED)
    d.text((x, 408), "data and my guesses written down first.", font=sub, fill=MUTED)
    d.text((x, 508), "Nishanth Rajamani  ·  nishzsche.github.io", font=font("regular", 24), fill=INK)
    img.save(OUT, optimize=True)
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
