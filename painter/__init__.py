"""auto_painter —— 自动化绘画器

把任意图片自动转成手绘风格的画作，支持：
- 自动风格识别（线稿 / 水墨 / 钢笔 / 素描 / 低多边形 / 照片）
- 多尺度边缘检测 + 距离变换线宽场
- KMeans 颜色量化 + 区域填色
- 变粗细贝塞尔平滑描边
- TSP 最近邻笔画排序（减少抬笔）
- 无头 PIL 直接渲染 PNG，同时导出可运行 turtle 脚本
"""

__version__ = "2.0.0"

from .pipeline import AutoPainter, PaintingResult
from .classifier import classify_image, STYLE_PRESETS

__all__ = ["AutoPainter", "PaintingResult", "classify_image", "STYLE_PRESETS"]
