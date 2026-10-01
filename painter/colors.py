"""颜色量化模块 —— 用 KMeans 提取主色调。"""

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def quantize_colors(img_bgr: np.ndarray, n_clusters: int = 10, random_state: int = 42):
    """对图片做颜色量化。

    Returns
    -------
    palette : (n_clusters, 3) RGB uint8
    quant_rgb : (h, w, 3) RGB uint8  量化后的图
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)

    km = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=random_state)
    labels = km.fit_predict(pixels)
    palette = km.cluster_centers_.astype(np.uint8)
    quant_rgb = palette[labels].reshape(h, w, 3)
    return palette, quant_rgb
