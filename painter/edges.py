"""边缘检测 + 线宽场计算。"""

import cv2
import numpy as np
from scipy.ndimage import sobel, median_filter


def _laplacian_sharp(gray: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.addWeighted(gray, 1.0, lap, 0.22, 0)


def preprocess_gray(img_bgr: np.ndarray) -> np.ndarray:
    """双边滤波保边平滑 → 背景差分 → CLAHE → 去噪 → 锐化。"""
    blur = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=60, sigmaSpace=60)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    bg = cv2.blur(gray, (28, 28))
    diff = cv2.absdiff(gray, bg)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(9, 9))
    enh = clahe.apply(diff)
    enh = cv2.fastNlMeansDenoising(enh, h=9.5, templateWindowSize=7, searchWindowSize=23)
    enh = _laplacian_sharp(enh)
    return enh


def multi_scale_edge(gray_enh: np.ndarray, low1: int = 18, high1: int = 130,
                     low2: int = 36, high2: int = 104) -> np.ndarray:
    """双尺度 Canny 合并 + 形态学清理。"""
    e1 = cv2.Canny(gray_enh, low1, high1, L2gradient=True)
    e2 = cv2.Canny(gray_enh, low2, high2, L2gradient=True)
    e2 = cv2.bitwise_and(e2, cv2.bitwise_not(e1))
    merged = cv2.bitwise_or(e1, e2)
    k_close = np.ones((3, 3), np.uint8)
    k_open = np.ones((2, 2), np.uint8)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, k_close)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, k_open)
    merged = median_filter(merged, size=2)
    return merged


def line_width_field(edge_map: np.ndarray, scale: float = 0.28,
                     w_min: float = 0.4, w_max: float = 6.0) -> np.ndarray:
    """距离变换生成线宽场：离边缘越远笔画越粗。"""
    dist = cv2.distanceTransform(255 - edge_map, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (5, 5), 1.2)
    field = dist * scale
    return np.clip(field, w_min, w_max)
