import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from implementation.image_processing import ImageProcessing
from PIL import Image
import numpy as np
import time

def test_methods():
    # Создаем экземпляр процессора
    processor = ImageProcessing()

    # Загружаем тестовое изображение (возьмем первое из test_images)
    image_path = "test_images/NotreDame.jpg"
    image = np.array(Image.open(image_path))

    # Проверяем каждый метод и замеряем время

    # 1. Преобразование в оттенки серого
    start_time = time.time()
    gray = processor._rgb_to_grayscale(image)
    gray_time = time.time() - start_time
    Image.fromarray(gray).save("gray_result.jpg")

    # 2. Гамма-коррекция
    start_time = time.time()
    gamma_corrected = processor._gamma_correction(image, 0.25)
    gamma_time = time.time() - start_time
    Image.fromarray(gamma_corrected).save("gamma_result.jpg")

    # 3. Обнаружение границ
    start_time = time.time()
    edges = processor.edge_detection(image)
    edges_time = time.time() - start_time
    Image.fromarray(edges).save("edges_result.jpg")

    # 4. Обнаружение углов
    start_time = time.time()
    corners = processor.corner_detection(image)
    corners_time = time.time() - start_time
    Image.fromarray(corners).save("corners_result.jpg")

    #start_time = time.time()
    #circles = processor.circle_detection(image)
    #circles_time = time.time() - start_time
    #Image.fromarray(circles).save("circles_result.jpg")

    # Выводим время выполнения
    print(f"Grayscale: {gray_time:.4f} сек.")
    print(f"Gamma: {gamma_time:.4f} сек.")
    print(f"Edges: {edges_time:.4f} сек.")
    print(f"Corners: {corners_time:.4f} сек.")
   # print(f"Circles: {circles_time:.4f} сек.")

if __name__ == "__main__":
    test_methods()