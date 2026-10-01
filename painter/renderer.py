"""渲染器：PIL 无头渲染 PNG + 导出 turtle 脚本。"""

import cv2
import numpy as np
from PIL import Image, ImageDraw


def render_pil(regions, strokes, width: int, height: int,
               bg_color=(255, 255, 255), line_width_field=None,
               stroke_color_mode="sampled", stroke_mono=(30, 30, 30)):
    """用 Pillow 直接渲染成图像（无需 GUI）。

    Parameters
    ----------
    regions : list[(poly_pts, rgb)]  要填充的闭合区域
    strokes : list[(pts, rgb)]  要描的开放笔画
    line_width_field : (h, w) array  每像素线宽
    """
    img = Image.new("RGB", (width, height), tuple(int(c) for c in bg_color))
    draw = ImageDraw.Draw(img)

    # 1) 区域填色
    for poly, rgb in regions:
        pts = [(float(x), float(y)) for x, y in poly]
        if len(pts) >= 3:
            draw.polygon(pts, fill=tuple(int(c) for c in rgb))

    # 2) 描边
    h, w = (line_width_field.shape[:2] if line_width_field is not None else (height, width))
    for pts, sampled_rgb in strokes:
        if len(pts) < 2:
            continue
        if stroke_color_mode == "mono":
            color = tuple(int(c) for c in stroke_mono)
        else:
            color = tuple(int(c) for c in sampled_rgb)

        # 逐段画线并根据线宽场调整粗细
        prev = None
        prev_w = None
        for p in pts:
            x, y = float(p[0]), float(p[1])
            if line_width_field is not None:
                xi = int(np.clip(x, 0, w - 1))
                yi = int(np.clip(y, 0, h - 1))
                lw = float(line_width_field[yi, xi])
            else:
                lw = 1.5
            if prev is not None:
                # 用圆头线段模拟可变粗细：画两段近似
                draw.line([prev, (x, y)], fill=color, width=max(1, int(round(lw))))
                r = max(0.5, lw / 2.0)
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
            prev = (x, y)
    return img


def build_regions(quant_rgb, palette, min_area: float = 80):
    """根据量化后的颜色图提取需要填充的区域。"""
    h, w = quant_rgb.shape[:2]
    regions = []
    for color in palette:
        mask = cv2.inRange(quant_rgb, color, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon=1.2, closed=True)
            if len(approx) < 3:
                continue
            pts = approx.reshape(-1, 2).astype(np.float64)
            regions.append((pts, tuple(int(c) for c in color)))
    regions.sort(key=lambda r: cv2.contourArea(r[0].reshape(-1, 1, 2).astype(np.float32)), reverse=True)
    return regions


def export_turtle_script(regions, strokes, width, height, path,
                         stroke_color_mode="sampled", stroke_mono=(30, 30, 30),
                         line_width_field=None):
    """导出可独立运行的 turtle 绘图脚本。"""
    off_x = -width / 2
    off_y = height / 2
    L = ["import turtle", "", "screen = turtle.Screen()",
         f"screen.setup({width + 100}, {height + 100})",
         "screen.tracer(0)",
         "pen = turtle.Turtle()", "pen.hideturtle()", "pen.speed(0)", "pen.penup()", ""]

    for poly, rgb in regions:
        trgb = tuple(c / 255.0 for c in rgb)
        sx, sy = poly[0]
        L.append("pen.penup()")
        L.append(f"pen.fillcolor{trgb}")
        L.append(f"pen.pencolor{trgb}")
        L.append(f"pen.goto({sx + off_x:.1f}, {-sy + off_y:.1f})")
        L.append("pen.pendown()")
        L.append("pen.begin_fill()")
        for px, py in poly[1:]:
            L.append(f"pen.goto({px + off_x:.1f}, {-py + off_y:.1f})")
        L.append(f"pen.goto({sx + off_x:.1f}, {-sy + off_y:.1f})")
        L.append("pen.end_fill()")
        L.append("pen.penup()")
        L.append("")

    h, w = (line_width_field.shape[:2] if line_width_field is not None else (height, width))
    for pts, sampled_rgb in strokes:
        if len(pts) < 2:
            continue
        trgb = tuple(c / 255.0 for c in (stroke_mono if stroke_color_mode == "mono" else sampled_rgb))
        x0, y0 = pts[0]
        if line_width_field is not None:
            xi = int(np.clip(x0, 0, w - 1)); yi = int(np.clip(y0, 0, h - 1))
            lw = float(line_width_field[yi, xi])
        else:
            lw = 1.5
        L.append("pen.penup()")
        L.append(f"pen.pencolor{trgb}")
        L.append(f"pen.pensize({lw:.2f})")
        L.append(f"pen.goto({x0 + off_x:.1f}, {-y0 + off_y:.1f})")
        L.append("pen.pendown()")
        for p in pts[1:]:
            x, y = float(p[0]), float(p[1])
            L.append(f"pen.goto({x + off_x:.1f}, {-y + off_y:.1f})")
        L.append("pen.penup()")
        L.append("")

    L += ["screen.update()", "turtle.done()", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
