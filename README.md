# 🎨 auto-painter · 自动化绘画器

把任意一张照片自动变成手绘风格的画作。基于 OpenCV 计算机视觉管线，自动完成颜色量化、边缘检测、变粗细描边与区域填色，**无需 GUI 即可直接渲染 PNG**，同时导出可独立运行的 Python turtle 绘图脚本。

## ✨ 特性

- 🤖 **自动风格识别** —— 自动判断线稿 / 水墨 / 钢笔 / 素描 / 低多边形 / 彩色照片，并套用对应参数
- 🎨 **KMeans 颜色量化** —— 提取主色调，按色块填充区域
- ✏️ **多尺度边缘检测** —— 双尺度 Canny 合并 + 形态学清理
- 🖌 **可变粗细描边** —— 距离变换生成线宽场，离边缘越远笔画越粗
- 🧭 **TSP 最近邻排序** —— 智能规划笔画顺序，减少抬笔
- 🖼 **无头渲染** —— Pillow 直接出 PNG，服务器 / 无显示器环境也能跑
- 🐢 **双输出** —— 同时导出可运行的 turtle 脚本，可在本地观看绘制过程

## 📦 安装

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

```bash
# 自动识别风格，输出到 output/
python main.py myphoto.jpg

# 指定风格
python main.py myphoto.jpg -s ink

# 自定义输出目录与尺寸
python main.py myphoto.jpg -s sketch -o result --max-side 1600
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `image` | 输入图片路径（必填） | — |
| `-s, --style` | 绘画风格：`auto`/`lineart`/`ink`/`pen`/`sketch`/`lowpoly`/`color` | `auto` |
| `-o, --out` | 输出目录 | `output` |
| `--scale` | 缩放系数 | `0.72` |
| `--max-side` | 输出最长边像素 | `1200` |
| `--no-png` | 不渲染 PNG | 关 |
| `--no-turtle` | 不导出 turtle 脚本 | 关 |

## 🎯 风格说明

| 风格 | 适用场景 | 笔触特点 |
|------|----------|----------|
| `lineart` | 线稿、漫画 | 干净黑边，少填色 |
| `ink` | 水墨、写意 | 粗笔刷，米白底色 |
| `pen` | 钢笔速写 | 细线条，高密度 |
| `sketch` | 铅笔素描 | 灰度采样，柔和 |
| `lowpoly` | 几何风 | 大色块，简化轮廓 |
| `color` | 彩色照片 | 彩色填色 + 彩色描边 |

## 🧩 作为库使用

```python
from painter import AutoPainter

painter = AutoPainter(style="auto")          # 或指定 "ink" / "pen" ...
result = painter.paint("myphoto.jpg", out_dir="output")

print(result.style, result.n_regions, result.n_strokes)
print(result.output_png, result.output_script)
```

## 🗂 项目结构

```
auto-painter/
├── main.py              # CLI 入口
├── requirements.txt
├── painter/
│   ├── __init__.py
│   ├── classifier.py    # 自动风格识别 + 预设
│   ├── colors.py        # KMeans 颜色量化
│   ├── edges.py         # 边缘检测 + 线宽场
│   ├── strokes.py       # 笔画提取/平滑/TSP排序
│   ├── renderer.py      # PIL 渲染 + turtle 脚本导出
│   └── pipeline.py      # 主管线
└── output/              # 生成结果（git 忽略）
```

## 📄 License

MIT
