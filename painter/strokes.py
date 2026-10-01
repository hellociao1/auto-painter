"""笔画提取、平滑与 TSP 排序。"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree


def smooth_points(pts: np.ndarray, sigma: float = 1.1) -> np.ndarray:
    """沿笔画做高斯平滑，保留端点。"""
    if len(pts) < 3:
        return pts
    pts = np.asarray(pts, dtype=np.float64)
    x = gaussian_filter1d(pts[:, 0], sigma=sigma)
    y = gaussian_filter1d(pts[:, 1], sigma=sigma)
    x[0], x[-1] = pts[0, 0], pts[-1, 0]
    y[0], y[-1] = pts[0, 1], pts[-1, 1]
    return np.column_stack((x, y))


def extract_strokes(edge_map: np.ndarray, img_rgb: np.ndarray,
                    min_area: float = 40, simplify_eps: float = 0.9,
                    smooth_sigma: float = 1.1):
    """从边缘图提取笔画，返回 [(pts, sampled_rgb), ...]。"""
    contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = edge_map.shape[:2]
    strokes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon=simplify_eps, closed=False)
        if len(approx) < 2:
            continue
        pts = approx.reshape(-1, 2).astype(np.float64)
        pts = smooth_points(pts, sigma=smooth_sigma)
        # 在笔画中点采样颜色
        mid = len(pts) // 2
        mx = int(np.clip(pts[mid, 0], 0, w - 1))
        my = int(np.clip(pts[mid, 1], 0, h - 1))
        r, g, b = img_rgb[my, mx]
        strokes.append((pts, (int(r), int(g), int(b))))
    return strokes


def tsp_order(strokes):
    """最近邻 TSP 排序笔画，减少抬笔移动。"""
    if len(strokes) <= 1:
        return strokes
    starts = np.array([s[0][0] for s in strokes], dtype=np.float64)
    kd = KDTree(starts)
    unvisited = set(range(len(strokes)))
    order = []
    cur = 0
    while unvisited:
        order.append(cur)
        unvisited.remove(cur)
        if not unvisited:
            break
        _, cands = kd.query(starts[cur], k=min(len(strokes), 20))
        for c in cands:
            c = int(c)
            if c in unvisited:
                cur = c
                break
    return [strokes[i] for i in order]
