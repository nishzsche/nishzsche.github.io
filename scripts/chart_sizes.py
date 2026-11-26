# /// script
# requires-python = ">=3.10"
# ///
"""Chart sizes for _includes/chart.html, so the page keeps their space before they load.

    uv run scripts/chart_sizes.py

Reads the width and height of every chart PNG under public/images/ (not the -dark versions,
which match their light ones, and not social cards) and writes _data/charts.yml. Run it after
adding or re-exporting charts.
"""

from __future__ import annotations

import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_data" / "charts.yml"


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"not a PNG: {path}")
    return struct.unpack(">II", head[16:24])


def main() -> None:
    lines = [
        "# Chart sizes in pixels, for _includes/chart.html. Written by scripts/chart_sizes.py:",
        "# run it again after adding or re-exporting charts.",
    ]
    for path in sorted((ROOT / "public" / "images").rglob("*.png")):
        if path.stem.endswith("-dark") or path.stem == "social-card":
            continue
        w, h = png_size(path)
        lines.append(f'"/{path.relative_to(ROOT).as_posix()}": [{w}, {h}]')
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}: {len(lines) - 2} charts")


if __name__ == "__main__":
    main()
