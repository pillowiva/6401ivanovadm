import os
import io
import asyncio
import aiohttp
import aiofiles
import numpy as np
from PIL import Image
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
import time
from dotenv import load_dotenv

from lab2.decorators.timer import timer_decorator
from lab2.images.color import ColorCatImage
from lab2.images.grayscale import GrayscaleCatImage

# Загрузка переменных окружения
load_dotenv()


def process_single_image(image_data: dict) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Обрабатывает одно изображение в отдельном процессе.

    Args:
        image_data: Данные изображения

    Returns:
        Кортеж (base_filename, custom_edges, library_edges)
    """
    index = image_data['index']
    breed = image_data['breed']
    image_array = image_data['image_array']
    url = image_data['url']

    print(f"Convolution for image {index} started (PID {os.getpid()})")

    # Создаем объект изображения
    if len(image_array.shape) == 3:
        cat_image = ColorCatImage(image_array, breed, url)
    else:
        cat_image = GrayscaleCatImage(image_array.astype(np.uint8), breed, url)

    # Обрабатываем изображение
    custom_edges = cat_image.detect_edges_custom()
    library_edges = cat_image.detect_edges_library()

    # Генерируем безопасное имя файла
    safe_breed = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in breed)
    base_filename = f"{index}_{safe_breed}"

    print(f"Convolution for image {index} finished (PID {os.getpid()})")

    return base_filename, custom_edges, library_edges


class AsyncCatImageProcessor:
    """
    Асинхронный класс для обработки изображений животных.

    Использует aiohttp для асинхронного скачивания и ProcessPoolExecutor
    для параллельной обработки изображений.
    """

    def __init__(self, api_url: str = "https://api.thecatapi.com/v1/images/search") -> None:
        """
        Инициализация процессора изображений.

        Аргументы:
            api_url: URL API для получения изображений
        """
        self._api_url = api_url
        self._api_key = os.getenv('API_KEY')
        self._image_data: List[dict] = []

    @property
    def image_data(self) -> List[dict]:
        """Property для получения данных изображений."""
        return self._image_data

    async def download_single_image(self, session: aiohttp.ClientSession,
                                    item, index: int, output_dir: str) -> None:
        """
        Асинхронно скачивает и сохраняет одно изображение.

        Args:
            session: aiohttp сессия
            item: Данные изображения из API
            index: Порядковый номер изображения
            output_dir: Директория для сохранения
        """
        print(f"Downloading image {index} started")


        try:
            # Скачиваем изображение асинхронно

            print(f"  Image {index} data keys: {item.keys()}")
            if 'breeds' in item:
                print(f"  Image {index} breeds: {item['breeds']}")
            async with session.get(item['url']) as response:
                if response.status == 200:
                    image_content = await response.read()

                    # Конвертируем в numpy array
                    image = Image.open(io.BytesIO(image_content))
                    image_array = np.array(image)
                    breed_name = 'unknown'

                    if 'breeds' in item and item['breeds'] and len(item['breeds']) > 0:
                        breed_info = item['breeds'][0]
                        breed_name = breed_info.get('name', 'unknown').replace(' ', '_').lower()

                    # Генерируем безопасное имя файла
                    safe_breed = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in breed_name)
                    original_filename = f"{index}_{safe_breed}_original.png"
                    original_path = os.path.join(output_dir, original_filename)

                    # Сохраняем оригинальное изображение асинхронно
                    async with aiofiles.open(original_path, 'wb') as f:
                        await f.write(image_content)

                    # Сохраняем данные для последующей обработки
                    self._image_data.append({
                        'index': index,
                        'breed': breed_name,
                        'url': item['url'],
                        'image_array': image_array,
                        'original_path': original_path
                    })

                    print(f"Downloading image {index} finished")
                else:
                    print(f"Error downloading image {index}: HTTP {response.status}")

        except Exception as e:
            print(f"Error downloading image {index}: {e}")

    @timer_decorator
    async def download_images_async(self, limit: int = 1, output_dir: str = "processed_images") -> None:
        """
        Асинхронно загружает изображения с API.

        Args:
            limit: Количество изображений для загрузки
            output_dir: Директория для сохранения

        Raises:
            Exception: При ошибках загрузки
        """
        print(f"Starting async download of {limit} images...")

        if not self._api_key:
            print("Error: API_KEY not found in environment variables")
            return

        # Создаем директорию если не существует
        os.makedirs(output_dir, exist_ok=True)

        params = {
            'limit': limit,
            'api_key': self._api_key,
            'has_breed': 1
        }

        try:
            async with aiohttp.ClientSession() as session:
                # Получаем список URL от API
                async with session.get(self._api_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                print(f"Received {len(data)} images from API")

                # Создаем задачи для скачивания каждого изображения
                tasks = []
                for i, item in enumerate(data):
                    # Порядковый номер определяется здесь и не меняется
                    task = self.download_single_image(session, item, i + 1, output_dir)
                    tasks.append(task)

                # Запускаем все задачи параллельно
                await asyncio.gather(*tasks)

                print(f"Successfully downloaded {len(self._image_data)} images")

        except Exception as e:
            print(f"Error during async download: {e}")
            raise

    @timer_decorator
    def process_images_parallel(self, output_dir: str = "processed_images") -> None:
        """
        Обрабатывает изображения параллельно в нескольких процессах.

        Args:
            output_dir: Директория для сохранения обработанных изображений
        """
        print("Starting parallel image processing...")

        if not self._image_data:
            print("No images to process")
            return

        # Сортируем данные по индексу для сохранения порядка
        sorted_data = sorted(self._image_data, key=lambda x: x['index'])

        # Используем ProcessPoolExecutor для параллельной обработки
        with ProcessPoolExecutor() as executor:
            # Отправляем задачи на выполнение
            futures = [executor.submit(process_single_image, data) for data in sorted_data]

            # Собираем результаты
            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Image {i + 1} processing completed")
                except Exception as e:
                    print(f"Error processing image {i + 1}: {e}")

        # Сохраняем обработанные изображения
        self._save_processed_images(results, output_dir)

        print("Parallel image processing finished")

    def _save_processed_images(self, results: List[Tuple], output_dir: str) -> None:
        """
        Сохраняет обработанные изображения.

        Args:
            results: Результаты обработки
            output_dir: Директория для сохранения
        """
        print("Saving processed images...")

        for base_filename, custom_edges, library_edges in results:
            try:
                custom_path = os.path.join(output_dir, f"{base_filename}_custom_edges.png")
                library_path = os.path.join(output_dir, f"{base_filename}_library_edges.png")

                # Сохраняем обработанные изображения
                Image.fromarray(custom_edges).save(custom_path)
                Image.fromarray(library_edges).save(library_path)

                print(f"Processed images saved: {base_filename}")

            except Exception as e:
                print(f"Error saving processed images for {base_filename}: {e}")

    def get_stats(self) -> dict:
        """
        Получение статистики по загруженным изображениям.

        Returns:
            Словарь со статистикой
        """
        stats = {
            'total_images': len(self._image_data),
            'color_images': 0,
            'grayscale_images': 0,
            'breeds': set(),
            'shapes': []
        }

        for data in self._image_data:
            stats['breeds'].add(data['breed'])
            stats['shapes'].append(data['image_array'].shape)

            if len(data['image_array'].shape) == 3:
                stats['color_images'] += 1
            else:
                stats['grayscale_images'] += 1

        stats['breeds'] = list(stats['breeds'])
        return stats


async def async_main(limit: int = 5, output_dir: str = "async_processed_images"):
    print("=== Async Cat Image Processor ===")

    # Создаем процессор
    processor = AsyncCatImageProcessor()

    try:
        # Замеряем общее время выполнения
        start_time = time.time()

        # Скачиваем изображения асинхронно
        await processor.download_images_async(limit=limit, output_dir=output_dir)

        # Обрабатываем изображения параллельно
        processor.process_images_parallel(output_dir=output_dir)

        # Выводим статистику
        stats = processor.get_stats()
        print("\n=== Statistics ===")
        print(f"Total images: {stats['total_images']}")
        print(f"Color images: {stats['color_images']}")
        print(f"Grayscale images: {stats['grayscale_images']}")
        print(f"Breeds: {stats['breeds']}")

        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

        return processor

    except Exception as e:
        print(f"Program execution failed: {e}")
        raise