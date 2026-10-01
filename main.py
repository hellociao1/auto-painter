#!/usr/bin/env python3
"""auto-painter 命令行入口。

用法:
    python main.py 输入图片 [-s 风格] [-o 输出目录] [--no-png] [--no-turtle]

风格可选: auto(默认) lineart ink pen sketch lowpoly color
"""

import argparse
import sys

from painter import AutoPainter, classify_image
import cv2


def main():
    ap = argparse.ArgumentParser(description="自动化绘画器 —— 把图片转成手绘风画作")
    ap.add_argument("image", help="输入图片路径")
    ap.add_argument("-s", "--style", default="auto",
                    choices=["auto", "lineart", "ink", "pen", "sketch", "lowpoly", "color"],
                    help="绘画风格 (默认 auto 自动识别)")
    ap.add_argument("-o", "--out", default="output", help="输出目录 (默认 output)")
    ap.add_argument("--scale", type=float, default=0.72, help="缩放系数")
    ap.add_argument("--max-side", type=int, default=1200, help="输出最长边像素")
    ap.add_argument("--no-png", action="store_true", help="不渲染 PNG")
    ap.add_argument("--no-turtle", action="store_true", help="不导出 turtle 脚本")
    args = ap.parse_args()

    style = None if args.style == "auto" else args.style

    # 先看看自动识别出来是什么风格
    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ 无法读取图片: {args.image}", file=sys.stderr)
        sys.exit(1)
    auto_style = classify_image(img)
    print(f"🔍 自动识别风格: {auto_style}" + ("" if style is None else f"  (强制使用: {style})"))

    painter = AutoPainter(style=style, scale=args.scale, max_side=args.max_side)
    result = painter.paint(
        args.image, out_dir=args.out,
        export_png=not args.no_png,
        export_turtle=not args.no_turtle,
    )

    print(f"🎨 绘画完成！风格 = {result.style}")
    print(f"   画布尺寸: {result.size[0]}×{result.size[1]}")
    print(f"   填色区域: {result.n_regions} 个")
    print(f"   描边笔画: {result.n_strokes} 条")
    if result.output_png:
        print(f"   🖼  PNG: {result.output_png}")
    if result.output_script:
        print(f"   🐢  Turtle 脚本: {result.output_script}")


if __name__ == "__main__":
    main()
