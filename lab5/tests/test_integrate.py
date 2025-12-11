# tests/test_live_api.py

import os
import shutil
import tempfile
import asyncio
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from cat_processor.processor.image_processor import AsyncCatImageProcessor


class TestLiveCatAPI(unittest.TestCase):
    """
    Интеграционный тест с реальными запросами к TheCatAPI.

    Запускается ТОЛЬКО если:
    - установлен API_KEY в переменных окружения
    - выставлен флаг RUN_LIVE_TESTS=1
    """

    @unittest.skipUnless(
        os.getenv("RUN_LIVE_TESTS") == "1",
        "LIVE-тесты выключены (RUN_LIVE_TESTS != 1)",
    )
    def test_real_api_download_and_process(self):
        api_key = os.getenv("API_KEY")
        if not api_key:
            self.skipTest("Нет API_KEY в окружении")

        tmpdir = Path(tempfile.mkdtemp(prefix="live_cat_api_"))

        try:
            processor = AsyncCatImageProcessor()

            # Скачиваем и обрабатываем реальные изображения
            asyncio.run(
                processor.download_images_async(limit=2, output_dir=str(tmpdir))
            )
            processor.process_images_parallel(output_dir=str(tmpdir))

            stats = processor.get_stats()

            # Минимальные здравые проверки
            self.assertEqual(stats["total_images"], 2)
            self.assertGreaterEqual(stats["color_images"], 1)

            files = list(tmpdir.iterdir())
            # хотя бы один original
            self.assertTrue(any("original" in f.name for f in files))
            # хотя бы один custom_edges
            self.assertTrue(any("custom_edges" in f.name for f in files))
            # хотя бы один library_edges
            self.assertTrue(any("library_edges" in f.name for f in files))

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


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


if __name__ == "__main__":
    unittest.main()
