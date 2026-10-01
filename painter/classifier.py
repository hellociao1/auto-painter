"""自动图像风格分类 + 预设参数。"""

import cv2
import numpy as np


# 每种风格的默认参数预设
STYLE_PRESETS = {
    "lineart": {
        "n_clusters": 6,
        "canny_low": 30, "canny_high": 140,
        "fill_min_area": 120,
        "stroke_min_area": 40,
        "line_width_scale": 0.22,
        "max_line_width": 5.5,
        "simplify_eps": 0.9,
        "smooth_sigma": 1.0,
        "bg_color": (255, 255, 255),
        "stroke_color_mode": "mono",   # mono / sampled
        "stroke_mono": (20, 20, 20),
    },
    "ink": {
        "n_clusters": 5,
        "canny_low": 20, "canny_high": 90,
        "fill_min_area": 200,
        "stroke_min_area": 60,
        "line_width_scale": 0.32,
        "max_line_width": 8.0,
        "simplify_eps": 1.4,
        "smooth_sigma": 1.6,
        "bg_color": (252, 250, 244),
        "stroke_color_mode": "mono",
        "stroke_mono": (25, 25, 28),
    },
    "pen": {
        "n_clusters": 8,
        "canny_low": 40, "canny_high": 160,
        "fill_min_area": 90,
        "stroke_min_area": 35,
        "line_width_scale": 0.18,
        "max_line_width": 3.2,
        "simplify_eps": 0.7,
        "smooth_sigma": 0.8,
        "bg_color": (255, 255, 255),
        "stroke_color_mode": "mono",
        "stroke_mono": (30, 30, 30),
    },
    "sketch": {
        "n_clusters": 7,
        "canny_low": 25, "canny_high": 110,
        "fill_min_area": 150,
        "stroke_min_area": 50,
        "line_width_scale": 0.20,
        "max_line_width": 4.0,
        "simplify_eps": 1.0,
        "smooth_sigma": 1.2,
        "bg_color": (250, 248, 240),
        "stroke_color_mode": "sampled",
        "stroke_mono": (60, 60, 60),
    },
    "lowpoly": {
        "n_clusters": 12,
        "canny_low": 50, "canny_high": 150,
        "fill_min_area": 60,
        "stroke_min_area": 25,
        "line_width_scale": 0.15,
        "max_line_width": 2.0,
        "simplify_eps": 2.0,
        "smooth_sigma": 0.5,
        "bg_color": (255, 255, 255),
        "stroke_color_mode": "mono",
        "stroke_mono": (80, 80, 80),
    },
    "color": {
        "n_clusters": 10,
        "canny_low": 18, "canny_high": 130,
        "fill_min_area": 80,
        "stroke_min_area": 45,
        "line_width_scale": 0.28,
        "max_line_width": 6.0,
        "simplify_eps": 1.0,
        "smooth_sigma": 1.1,
        "bg_color": (255, 255, 255),
        "stroke_color_mode": "sampled",
        "stroke_mono": (40, 40, 40),
    },
}


def classify_image(img_bgr: np.ndarray) -> str:
    """根据图像统计特征自动判断最合适的绘画风格。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges_raw = cv2.Canny(gray, 40, 120)
    edge_density = np.count_nonzero(edges_raw) / gray.size

    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist /= hist.sum()
    peak_idx = int(np.argmax(hist))

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(hsv[..., 1].mean())
    color_std = float(img_bgr.std(axis=(0, 1)).mean())
    canny_var = float(cv2.Laplacian(edges_raw, cv2.CV_64F).var())

    if edge_density > 0.12 and color_std < 48 and sat_mean < 62 and canny_var > 1200:
        return "lineart"
    if lap_var < 260 and edge_density < 0.092 and sat_mean < 53 and peak_idx > 13:
        return "ink"
    if edge_density > 0.132 and lap_var > 520 and sat_mean < 68:
        return "pen"
    if 0.062 < edge_density <= 0.142 and 220 < lap_var < 720 and sat_mean < 72:
        return "sketch"
    if color_std < 63 and 11 < peak_idx < 23 and sat_mean > 32:
        return "lowpoly"
    return "color"
