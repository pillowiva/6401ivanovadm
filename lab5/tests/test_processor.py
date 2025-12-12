import unittest
from unittest.mock import Mock, patch

import numpy as np
from pathlib import Path
import os
import tempfile
import shutil
from cat_processor import AsyncCatImageProcessor, process_single_image

class TestAsyncCatImageProcessor(unittest.TestCase):
    """3 основных теста для класса AsyncCatImageProcessor."""

    def setUp(self):
        """Подготовка тестовых данных."""
        self.processor = AsyncCatImageProcessor()

        # Создаем тестовый массив
        self.test_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Создаем временную директорию для тестов
        self.test_dir = tempfile.mkdtemp(prefix="test_output_")
        print(f"Создана тестовая директория: {self.test_dir}")

    def tearDown(self):
        """Очистка после тестов."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"Удалена тестовая директория: {self.test_dir}")

    def test_1_processor_initialization_and_basic_functionality(self):
        """Тест 1: Инициализация и базовый функционал процессора."""
        # Проверяем создание объекта
        self.assertIsInstance(self.processor, AsyncCatImageProcessor)

        # Проверяем атрибуты по умолчанию
        self.assertEqual(self.processor._api_url, "https://api.thecatapi.com/v1/images/search")
        self.assertEqual(self.processor._image_data, [])

        # Проверяем свойство image_data
        self.assertEqual(self.processor.image_data, [])

        # Тестируем get_stats с пустыми данными
        stats = self.processor.get_stats()
        self.assertEqual(stats['total_images'], 0)
        self.assertEqual(stats['color_images'], 0)
        self.assertEqual(stats['grayscale_images'], 0)

        print("✓ Тест 1 пройден: инициализация и базовый функционал работают")

    def test_2_statistics_with_test_data(self):
        """Тест 2: Проверка статистики с тестовыми данными."""
        # Добавляем тестовые данные напрямую
        self.processor._image_data = [
            {
                'breed': 'Breed1',
                'image_array': np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            },
            {
                'breed': 'Breed2',
                'image_array': np.random.randint(0, 256, (150, 150), dtype=np.uint8)
            }
        ]

        # Получаем статистику
        stats = self.processor.get_stats()

        # Проверяем статистику
        self.assertEqual(stats['total_images'], 2)
        self.assertEqual(stats['color_images'], 1)
        self.assertEqual(stats['grayscale_images'], 1)
        self.assertEqual(len(stats['breeds']), 2)
        self.assertIn('Breed1', stats['breeds'])
        self.assertIn('Breed2', stats['breeds'])

        print("✓ Тест 2 пройден: статистика корректно рассчитывается")

    def test_3_process_single_image_function(self):
        """Тест 3: Функция обработки одного изображения."""
        # Тестовые данные
        image_data = {
            'index': 1,
            'breed': 'TestBreed',
            'image_array': self.test_array,
            'url': 'http://test.com/cat.jpg'
        }

        # Вызываем функцию
        base_filename, custom_edges, library_edges = process_single_image(image_data)

        # Проверяем результаты
        self.assertIsInstance(base_filename, str)
        self.assertIn('1_TestBreed', base_filename)

        # Проверяем, что возвращаются массивы правильной формы
        self.assertIsInstance(custom_edges, np.ndarray)
        self.assertIsInstance(library_edges, np.ndarray)
        self.assertEqual(custom_edges.shape, (100, 100))
        self.assertEqual(library_edges.shape, (100, 100))

        # Проверяем, что результаты не пустые
        self.assertFalse(np.all(custom_edges == 0))
        self.assertFalse(np.all(library_edges == 0))

        print("✓ Тест 3 пройден: функция обработки изображения работает")

class DummyFuture:
    """Простой Future, который сразу содержит результат.

    Нужен, чтобы подменить поведение ProcessPoolExecutor так,
    чтобы задачи выполнялись в том же процессе (без fork).
    """

    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class DummyExecutor:
    """Заглушка вместо ProcessPoolExecutor для тестов.

    Поддерживает контекстный менеджер и метод submit, который
    сразу выполняет функцию и заворачивает результат в DummyFuture.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Ничего особенного при выходе из контекста делать не нужно
        return False

    def submit(self, fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        return DummyFuture(result)


class TestProcessorWithMockCatImage(unittest.TestCase):
    """
    Интеграционный тест AsyncCatImageProcessor с использованием Mock-CatImage.

    В этом тесте:
    * не ходим в сеть;
    * не создаём настоящие ColorCatImage / GrayscaleCatImage;
    * подменяем их Mock-объектами;
    * запускаем реальный process_images_parallel.
    """

    def setUp(self):
        # Временная директория для сохранения результатов обработки
        self.tmpdir = Path(tempfile.mkdtemp(prefix="mock_catimage_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("cat_processor.processor.image_processor.ProcessPoolExecutor", DummyExecutor)
    @patch("cat_processor.processor.image_processor.GrayscaleCatImage")
    @patch("cat_processor.processor.image_processor.ColorCatImage")
    def test_process_images_with_mock_catimage(self, mock_color_cls, mock_gray_cls):
        # 1. Готовим исходные данные: одно цветное изображение
        processor = AsyncCatImageProcessor()

        image_array = np.zeros((10, 10, 3), dtype=np.uint8)
        breed = "mock_breed"
        url = "https://example.com/mock.png"

        processor._image_data = [
            {
                "index": 1,
                "breed": breed,
                "url": url,
                "image_array": image_array,
                # путь оригинала нам тут особо не важен, но добавим для реалистичности
                "original_path": str(self.tmpdir / "1_mock_breed_original.png"),
            }
        ]

        # 2. Настраиваем Mock-CatImage
        mock_cat_instance = Mock()
        mock_cat_instance.detect_edges_custom.return_value = np.ones(
            (10, 10), dtype=np.uint8
        )
        mock_cat_instance.detect_edges_library.return_value = np.full(
            (10, 10), 2, dtype=np.uint8
        )

        # При создании ColorCatImage(...) должен возвращаться наш mock-объект
        mock_color_cls.return_value = mock_cat_instance

        # 3. Запускаем реальную обработку (но без реальноого ProcessPoolExecutor)
        processor.process_images_parallel(output_dir=str(self.tmpdir))

        # 4. Проверяем, что использовался именно ColorCatImage, а не GrayscaleCatImage
        mock_color_cls.assert_called_once()
        mock_gray_cls.assert_not_called()

        # Конструктор ColorCatImage должен был получить исходные данные
        args, kwargs = mock_color_cls.call_args
        self.assertIs(args[0], image_array)
        self.assertEqual(args[1], breed)
        self.assertEqual(args[2], url)

        # Методы Mock-CatImage должны были быть вызваны при обработке
        mock_cat_instance.detect_edges_custom.assert_called_once()
        mock_cat_instance.detect_edges_library.assert_called_once()

        # 5. Проверяем, что результаты обработки сохранились в файлы
        custom_path = self.tmpdir / "1_mock_breed_custom_edges.png"
        library_path = self.tmpdir / "1_mock_breed_library_edges.png"

        self.assertTrue(
            custom_path.exists(), "Файл с пользовательскими контурами не создан"
        )
        self.assertTrue(
            library_path.exists(), "Файл с библиотечными контурами не создан"
        )

        # И файлы не пустые
        self.assertGreater(custom_path.stat().st_size, 0)
        self.assertGreater(library_path.stat().st_size, 0)

