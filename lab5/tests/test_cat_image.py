import unittest
import numpy as np

from cat_processor import ColorCatImage, GrayscaleCatImage
from utils import ImageProcessing

class TestCatImage(unittest.TestCase):
    """Тесты для класса CatImage и его наследников."""

    def setUp(self):
        """Подготовка тестовых данных перед каждым тестом."""
        # Создаем тестовые изображения
        self.color_array1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.color_array2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.gray_array1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        self.gray_array2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        # Создаем объекты изображений
        self.color_image1 = ColorCatImage(
            self.color_array1,
            "TestBreed1",
            "http://test1.com"
        )
        self.color_image2 = ColorCatImage(
            self.color_array2,
            "TestBreed2",
            "http://test2.com"
        )
        self.gray_image1 = GrayscaleCatImage(
            self.gray_array1,
            "TestBreed1",
            "http://test1.com"
        )
        self.gray_image2 = GrayscaleCatImage(
            self.gray_array2,
            "TestBreed2",
            "http://test2.com"
        )

    def tearDown(self):
        """Очистка после каждого теста."""
        pass

    def test_rgb_to_grayscale_conversion(self):
        """Тест конвертации RGB в оттенки серого."""
        proc = ImageProcessing()

        # Получаем массив из объекта ColorCatImage
        color_array = self.color_image1.image_array
        gray = proc._rgb_to_grayscale(color_array)

        self.assertEqual(gray.ndim, 2)  # Должна быть 2D матрица
        self.assertEqual(gray.shape, (100, 100))  # Размер должен сохраниться
        self.assertEqual(gray.dtype, np.uint8)
        self.assertTrue(gray.min() >= 0 and gray.max() <= 255)

        # Проверка формулы конвертации
        # Y = 0.299*R + 0.587*G + 0.114*B
        test_pixel = color_array[0, 0]
        expected_value = np.dot(test_pixel[:3], [0.299, 0.587, 0.114])
        actual_value = gray[0, 0]

        # Допускаем небольшую погрешность из-за округления
        self.assertAlmostEqual(actual_value, expected_value, delta=1)

    def test_convolution_operation(self):
        """Тест операции свертки."""
        # Создаем тестовое ядро
        kernel = np.array([[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1]])

        proc = ImageProcessing()
        conv_result_gray = proc._convolution(self.gray_image1._image_array, kernel)
        self.assertEqual(conv_result_gray.shape, (100, 100))

    def test_addition_color_images(self):
        """Тест сложения цветных изображений."""
        result = self.color_image1 + self.color_image2
        self.assertIsInstance(result, ColorCatImage)
        self.assertEqual(result.breed, "TestBreed1_added")
        self.assertEqual(result.shape[:2], (100, 100))

    def test_addition_grayscale_images(self):
        """Тест сложения ч/б изображений."""
        result = self.gray_image1 + self.gray_image2
        self.assertIsInstance(result, GrayscaleCatImage)
        self.assertEqual(result.breed, "TestBreed1_added")
        self.assertEqual(result.shape, (100, 100))