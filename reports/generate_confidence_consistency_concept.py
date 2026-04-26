#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    from reportlab.lib.pagesizes import landscape
    from reportlab.pdfgen import canvas
except Exception:  # pragma: no cover - PDF output is optional.
    canvas = None
    landscape = None


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figs"
OUT_STEM = OUT_DIR / "confidence_consistency_concept"

W, H = 3600, 1600
MARGIN_X = 140
PANEL_GAP = 90
PANEL_W = (W - 2 * MARGIN_X - 2 * PANEL_GAP) // 3
TOP_Y, TOP_H = 250, 650
BAR_Y, BAR_H = 970, 445


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


F_TITLE = font(54, bold=True)
F_PANEL = font(40, bold=True)
F_SUB = font(30)
F_SMALL = font(25)
F_TINY = font(21)
F_BAR = font(28, bold=True)


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, ft: ImageFont.ImageFont, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=ft)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=ft, fill=fill)


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color: str, width: int = 5) -> None:
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_len = 26
    spread = 0.55
    p1 = (
        end[0] - head_len * math.cos(angle - spread),
        end[1] - head_len * math.sin(angle - spread),
    )
    p2 = (
        end[0] - head_len * math.cos(angle + spread),
        end[1] - head_len * math.sin(angle + spread),
    )
    draw.polygon([end, p1, p2], fill=color)


def interp(panel: tuple[int, int, int, int], p: tuple[float, float]) -> tuple[float, float]:
    x, y, w, h = panel
    return x + p[0] * w, y + p[1] * h


def draw_star(draw: ImageDraw.ImageDraw, center: tuple[float, float], radius: float, fill: str) -> None:
    points = []
    for i in range(10):
        r = radius if i % 2 == 0 else radius * 0.43
        a = -math.pi / 2 + i * math.pi / 5
        points.append((center[0] + r * math.cos(a), center[1] + r * math.sin(a)))
    draw.polygon(points, fill=fill)


def draw_feature_panel(draw: ImageDraw.ImageDraw, col: int, kind: str) -> None:
    x0 = MARGIN_X + col * (PANEL_W + PANEL_GAP)
    draw.rounded_rectangle([x0, TOP_Y, x0 + PANEL_W, TOP_Y + TOP_H], radius=34, fill="#f8fafc", outline="#cbd5e1", width=3)

    inner = (x0 + 52, TOP_Y + 118, PANEL_W - 104, TOP_H - 190)
    centers = {
        "A": (0.23, 0.25),
        "B": (0.76, 0.27),
        "C": (0.28, 0.76),
        "D": (0.75, 0.75),
    }
    colors = {"A": "#2563eb", "B": "#dc2626", "C": "#16a34a", "D": "#9333ea"}
    fills = {"A": "#dbeafe", "B": "#fee2e2", "C": "#dcfce7", "D": "#f3e8ff"}

    for label, p in centers.items():
        cx, cy = interp(inner, p)
        draw.ellipse([cx - 84, cy - 84, cx + 84, cy + 84], fill=fills[label], outline=colors[label], width=4)
        draw_star(draw, (cx, cy), 42, colors[label])
        text_center(draw, (cx, cy + 105), f"ID {label}", F_TINY, colors[label])

    if kind == "normal":
        points = [(0.245, 0.32), (0.275, 0.28), (0.235, 0.24), (0.265, 0.35)]
        nearest_key = "A"
        edge, fill = "#1d4ed8", "#60a5fa"
        title = "(1) normal ID"
        subtitle = "confidence high / consistency high"
        note = "stable nearest class"
    elif kind == "ood":
        points = [(0.57, 0.31), (0.73, 0.43), (0.52, 0.55), (0.83, 0.36)]
        nearest_key = "B"
        edge, fill = "#d97706", "#fbbf24"
        title = "(2) near-class OOD"
        subtitle = "confidence high / consistency low"
        note = "nearest class changes"
    else:
        points = [(0.40, 0.42), (0.48, 0.50), (0.52, 0.39), (0.43, 0.57)]
        nearest_key = "A"
        edge, fill = "#475569", "#94a3b8"
        title = "(3) hard ID"
        subtitle = "consistency lower; confidence helps"
        note = "ambiguous but still close"

    px = [interp(inner, p)[0] for p in points]
    py = [interp(inner, p)[1] for p in points]
    sample = (sum(px) / len(px), sum(py) / len(py))
    nearest = interp(inner, centers[nearest_key])

    draw_arrow(draw, sample, nearest, edge, width=5)
    for point in zip(px, py):
        draw.line([sample, point], fill=edge, width=3)
    for point in zip(px, py):
        draw.ellipse([point[0] - 22, point[1] - 22, point[0] + 22, point[1] + 22], fill=fill, outline=edge, width=3)
    draw.ellipse([sample[0] - 36, sample[1] - 36, sample[0] + 36, sample[1] + 36], fill=edge, outline="white", width=5)

    draw.text((x0 + 34, TOP_Y + 28), title, font=F_PANEL, fill="#0f172a")
    draw.text((x0 + 34, TOP_Y + 82), subtitle, font=F_SUB, fill="#334155")
    text_center(draw, (x0 + PANEL_W / 2, TOP_Y + TOP_H - 42), note, F_SMALL, "#334155")


def draw_bar_panel(draw: ImageDraw.ImageDraw, col: int, kind: str) -> None:
    x0 = MARGIN_X + col * (PANEL_W + PANEL_GAP)
    draw.rounded_rectangle([x0, BAR_Y, x0 + PANEL_W, BAR_Y + BAR_H], radius=28, fill="#ffffff", outline="#cbd5e1", width=3)

    if kind == "normal":
        vals, errs, color, c_text, s_text = [0.15, 0.78, 0.70, 0.86], [0.03, 0.04, 0.04, 0.03], "#3b82f6", "C: high", "S: high"
    elif kind == "ood":
        vals, errs, color, c_text, s_text = [0.48, 0.20, 0.42, 0.35], [0.15, 0.18, 0.16, 0.19], "#f59e0b", "C: high", "S: low"
    else:
        vals, errs, color, c_text, s_text = [0.32, 0.45, 0.52, 0.66], [0.10, 0.12, 0.10, 0.09], "#64748b", "C: medium-high", "S: medium"

    plot_x, plot_y = x0 + 100, BAR_Y + 66
    plot_w, plot_h = PANEL_W - 155, BAR_H - 145
    axis_color = "#64748b"
    grid_color = "#e5e7eb"
    draw.line([(plot_x, plot_y), (plot_x, plot_y + plot_h), (plot_x + plot_w, plot_y + plot_h)], fill=axis_color, width=3)
    for k in range(1, 5):
        gy = plot_y + plot_h - k * plot_h / 5
        draw.line([(plot_x, gy), (plot_x + plot_w, gy)], fill=grid_color, width=2)

    bar_w = plot_w / 7.0
    labels = ["A", "B", "C", "D"]
    for i, (val, err, label) in enumerate(zip(vals, errs, labels)):
        cx = plot_x + (i + 0.75) * plot_w / 4
        y_top = plot_y + plot_h * (1 - val)
        y_err_low = plot_y + plot_h * (1 - min(1.0, val + err))
        y_err_high = plot_y + plot_h * (1 - max(0.0, val - err))
        draw.rounded_rectangle([cx - bar_w / 2, y_top, cx + bar_w / 2, plot_y + plot_h], radius=6, fill=color, outline="#1f2937", width=2)
        draw.line([(cx, y_err_low), (cx, y_err_high)], fill="#111827", width=3)
        draw.line([(cx - 18, y_err_low), (cx + 18, y_err_low)], fill="#111827", width=3)
        draw.line([(cx - 18, y_err_high), (cx + 18, y_err_high)], fill="#111827", width=3)
        text_center(draw, (cx, plot_y + plot_h + 38), label, F_SMALL, "#334155")

    draw.text((x0 + 35, BAR_Y + 26), c_text, font=F_BAR, fill="#0f172a")
    draw.text((x0 + 35, BAR_Y + 70), s_text, font=F_BAR, fill="#0f172a")
    draw.text((x0 + PANEL_W - 245, BAR_Y + 32), "lower = closer", font=F_TINY, fill="#475569")
    draw.text((x0 + 23, plot_y + 72), "distance", font=F_TINY, fill="#475569")


def svg_copy() -> str:
    png_name = f"{OUT_STEM.name}.png"
    # Keep the SVG simple and robust: embed the PNG as a data URI.
    import base64

    payload = base64.b64encode((OUT_STEM.with_suffix(".png")).read_bytes()).decode("ascii")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
        f'  <image width="{W}" height="{H}" href="data:image/png;base64,{payload}"/>\n'
        f'  <!-- Raster-backed SVG companion for {png_name}. -->\n'
        "</svg>\n"
    )


def write_pdf_from_png() -> None:
    if canvas is None or landscape is None:
        return
    png_path = OUT_STEM.with_suffix(".png")
    pdf_path = OUT_STEM.with_suffix(".pdf")
    page_w, page_h = landscape((W, H))
    c = canvas.Canvas(str(pdf_path), pagesize=(page_w, page_h))
    c.drawImage(str(png_path), 0, 0, width=page_w, height=page_h)
    c.showPage()
    c.save()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), "#ffffff")
    draw = ImageDraw.Draw(img)

    text_center(draw, (W / 2, 72), "Roles of confidence and consistency in MAF", F_TITLE, "#0f172a")
    draw.text(
        (MARGIN_X, 142),
        "Confidence captures closeness to the nearest ID class; consistency checks whether the distance structure remains stable.",
        font=F_SUB,
        fill="#334155",
    )

    for col, kind in enumerate(["normal", "ood", "hard"]):
        draw_feature_panel(draw, col, kind)
        draw_bar_panel(draw, col, kind)

    draw.text(
        (MARGIN_X, H - 82),
        "Distance bars show distances to ID prototypes A-D. Error bars illustrate instability across local perturbations/views.",
        font=F_SMALL,
        fill="#334155",
    )

    img.save(OUT_STEM.with_suffix(".png"), dpi=(300, 300))
    write_pdf_from_png()
    OUT_STEM.with_suffix(".svg").write_text(svg_copy(), encoding="utf-8")

    print(OUT_STEM.with_suffix(".png"))
    print(OUT_STEM.with_suffix(".pdf"))
    print(OUT_STEM.with_suffix(".svg"))


if __name__ == "__main__":
    main()
