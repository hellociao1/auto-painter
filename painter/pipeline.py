"""主绘画管线：把图片一键变成手绘风画作。"""

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .classifier import classify_image, STYLE_PRESETS
from .colors import quantize_colors
from .edges import preprocess_gray, multi_scale_edge, line_width_field
from .strokes import extract_strokes, tsp_order
from .renderer import render_pil, build_regions, export_turtle_script


@dataclass
class PaintingResult:
    style: str
    output_png: Optional[str] = None
    output_script: Optional[str] = None
    n_regions: int = 0
    n_strokes: int = 0
    size: tuple = (0, 0)


class AutoPainter:
    """自动化绘画器。

    Parameters
    ----------
    style : str or None
        绘画风格。None 表示自动识别。可选：
        lineart / ink / pen / sketch / lowpoly / color
    scale : float
        整体缩放系数。
    max_side : int
        输出图片最长边像素。
    """

    def __init__(self, style: Optional[str] = None, scale: float = 0.72,
                 max_side: int = 1200, **overrides):
        self.style_arg = style
        self.scale = scale
        self.max_side = max_side
        self.overrides = overrides

    def _resolve_preset(self, img_bgr) -> dict:
        if self.style_arg and self.style_arg in STYLE_PRESETS:
            name = self.style_arg
        else:
            name = classify_image(img_bgr)
        preset = dict(STYLE_PRESETS[name])
        preset.update({k: v for k, v in self.overrides.items() if v is not None})
        return name, preset

    def paint(self, image_path: str, out_dir: str = "output",
              export_png: bool = True, export_turtle: bool = True) -> PaintingResult:
        img_raw = cv2.imread(image_path)
        if img_raw is None:
            raise FileNotFoundError(f"图片读取失败: {image_path}")

        h0, w0 = img_raw.shape[:2]
        ratio = min(self.max_side / max(w0, h0), 1.0) * self.scale
        tw, th = max(256, int(w0 * ratio)), max(256, int(h0 * ratio))
        img = cv2.resize(img_raw, (tw, th), interpolation=cv2.INTER_LANCZOS4)

        style_name, p = self._resolve_preset(img)

        # 颜色量化 + 区域
        palette, quant_rgb = quantize_colors(img, n_clusters=p["n_clusters"])
        regions = build_regions(quant_rgb, palette, min_area=p["fill_min_area"])

        # 边缘 + 线宽场
        gray_enh = preprocess_gray(img)
        edge_map = multi_scale_edge(gray_enh, p["canny_low"], p["canny_high"])
        lw_field = line_width_field(edge_map, scale=p["line_width_scale"],
                                    w_min=0.4, w_max=p["max_line_width"])

        # 笔画
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        strokes = extract_strokes(edge_map, img_rgb,
                                  min_area=p["stroke_min_area"],
                                  simplify_eps=p["simplify_eps"],
                                  smooth_sigma=p["smooth_sigma"])
        strokes = tsp_order(strokes)

        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        result = PaintingResult(style=style_name, size=(tw, th),
                                n_regions=len(regions), n_strokes=len(strokes))

        if export_png:
            out_png = os.path.join(out_dir, f"{base}_painted.png")
            pil_img = render_pil(regions, strokes, tw, th,
                                 bg_color=p["bg_color"],
                                 line_width_field=lw_field,
                                 stroke_color_mode=p["stroke_color_mode"],
                                 stroke_mono=p["stroke_mono"])
            pil_img.save(out_png)
            result.output_png = out_png

        if export_turtle:
            out_py = os.path.join(out_dir, f"{base}_turtle.py")
            export_turtle_script(regions, strokes, tw, th, out_py,
                                 stroke_color_mode=p["stroke_color_mode"],
                                 stroke_mono=p["stroke_mono"],
                                 line_width_field=lw_field)
            result.output_script = out_py

        return result
