import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from lab1.implementation import ImageProcessing
import argparse, time
from PIL import Image
import numpy as np
import asyncio

# Добавляем импорт асинхронного процессора
# Предполагается, что файл с AsyncCatImageProcessor называется async_processor.py
from lab4.processor.image_processor import AsyncCatImageProcessor, async_main


def load(path):
    arr = np.array(Image.open(path).convert("RGB"))
    return arr


def save(path, arr):
    Image.fromarray(arr).save(path)


def default_name(path, op):
    import os
    base, ext = os.path.splitext(path)
    return f"{base}_{op}.png"  # Исправлено: убрана константа 2


def main():
    p = argparse.ArgumentParser(prog="imageprocessing")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("grayscale")
    g = sub.add_parser("gamma");
    g.add_argument("--value", type=float, required=True)
    sub.add_parser("sobel")
    h = sub.add_parser("harris");
    h.add_argument("--k", type=float, default=0.04);
    h.add_argument("--thr", type=float, default=0.01)
    c = sub.add_parser("hough");
    c.add_argument("--min-radius", type=int, default=10);
    c.add_argument("--max-radius", type=int);
    c.add_argument("--step", type=int, default=5)

    # Добавляем команду для свертки
    conv = sub.add_parser("convolution")
    conv.add_argument("--kernel", type=str, default="gaussian",
                      choices=["blur", "gaussian", "sharpen", "edge_detect",
                               "emboss", "sobel_x", "sobel_y", "laplacian",
                               "box_blur", "motion_blur"],
                      help="Тип ядра для свертки")

    # ДОБАВЛЯЕМ НОВУЮ КОМАНДУ ДЛЯ АСИНХРОННОЙ ОБРАБОТКИ КОТИКОВ
    async_cmd = sub.add_parser("async_cats")
    async_cmd.add_argument("--limit", type=int, default=5,
                           help="Количество изображений для загрузки")
    async_cmd.add_argument("--output-dir", type=str, default="async_cats_processed",
                           help="Директория для сохранения")

    p.add_argument("input_path", nargs="?")  # Делаем необязательным для async_cats
    p.add_argument("output_path", nargs="?")

    args = p.parse_args()

    # Для команды async_cats не нужен input_path
    if args.cmd == "async_cats":
        print("Запуск асинхронной обработки изображений котиков...")
        asyncio.run(async_main(limit=args.limit, output_dir=args.output_dir))
        return

    # Для остальных команд проверяем наличие input_path
    if not hasattr(args, 'input_path') or not args.input_path:
        print("Ошибка: для этой команды требуется input_path")
        return

    img = load(args.input_path)
    proc = ImageProcessing()

    t0 = time.perf_counter()
    if args.cmd == "grayscale":
        out = proc._rgb_to_grayscale(img)
    elif args.cmd == "gamma":
        out = proc._gamma_correction(img, args.value)
    elif args.cmd == "sobel":
        out = proc.edge_detection(img)
    elif args.cmd == "harris":
        out = proc.corner_detection(img, k=args.k, threshold=args.thr)
    elif args.cmd == "hough":
        if not args.max_radius:
            hgt, wdt = (img.shape[0], img.shape[1]) if img.ndim == 2 else (img.shape[0], img.shape[1])
            args.max_radius = min(hgt, wdt) // 4
        out = proc.circle_detection(img, min_radius=args.min_radius, max_radius=args.max_radius, step=args.step)
    elif args.cmd == "convolution":
        out = proc.apply_color_convolution(img, args.kernel)
    dt = time.perf_counter() - t0
    print(f"elapsed_sec={dt:.3f}")

    out_path = args.output_path or default_name(args.input_path, args.cmd)
    save(out_path, out)


if __name__ == "__main__":
    main()