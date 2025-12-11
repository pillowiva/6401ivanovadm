import os
import io
import requests
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Any
from dotenv import load_dotenv

from lab2.decorators.timer import timer_decorator
from lab2.images.color import ColorCatImage
from lab2.images.grayscale import GrayscaleCatImage


# Загрузка переменных окружения
load_dotenv()


class CatImageProcessor:
    """
    Класс для обработки изображений животных.

    Инкапсулирует функционал работы с API, управляет процессом
    обработки и сохранения скаченных изображений.

    Атрибуты:
        _api_url (str): URL API для получения изображений
        _api_key (str): API ключ для аутентификации
        _downloaded_images (List): Список загруженных изображений
    """

    def __init__(self, api_url: str = "https://api.thecatapi.com/v1/images/search") -> None:
        """
        Инициализация процессора изображений.

        Аргументы:
            api_url: URL API для получения изображений
        """
        self._api_url = api_url
        self._api_key = os.getenv('API_KEY')
        self._downloaded_images: List = []

    @property
    def downloaded_images(self) -> List:
        """Property для получения списка загруженных изображений."""
        return self._downloaded_images

    @downloaded_images.setter
    def downloaded_images(self, value: List) -> None:
        self._downloaded_images.append(value)

    @property
    def api_key(self) -> Optional[str]:
        """Property для получения API ключа (только для чтения)."""
        return self._api_key

    @timer_decorator
    def download_images(self, limit: int = 1) -> None:
        """
        Загрузка изображений с API.

        Args:
            limit: Количество изображений для загрузки

        Raises:
            requests.exceptions.RequestException: При ошибках сетевого запроса
            Exception: При других неожиданных ошибках
        """
        print(f"Начинаем загрузку {limit} изображений...")

        if not self._api_key:
            print("Ошибка: API_KEY не найден в переменных окружения")
            return

        params = {
            'limit': limit,

            'api_key': self._api_key
        }

        try:
            response = requests.get(self._api_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            print(f"Получено {len(data)} изображений от API")

            for i, item in enumerate(data):
                print(f"Обрабатываем изображение {i + 1}...")

                # Загружаем изображение
                image_response = requests.get(item['url'], timeout=30)
                image_response.raise_for_status()

                # Конвертируем в numpy array
                image = Image.open(io.BytesIO(image_response.content))
                image_array = np.array(image)

                # Получаем информацию о породе
                breed_info = item.get('breeds', [{}])[0]
                breed_name = breed_info.get('name', 'unknown').replace(' ', '_').lower()

                # Определяем тип изображения и создаем соответствующий объект
                cat_image = self._create_image_object(image_array, breed_name, item['url'])

                self._downloaded_images.append(cat_image)
                print(f"Изображение {i + 1} ('{breed_name}') успешно загружено")

        except requests.exceptions.RequestException as e:
            print(f"Ошибка при загрузке изображений: {e}")
            raise
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")
            raise

    def _create_image_object(self, image_array: np.ndarray, breed: str, url: str) -> Any:
        """
        Создает объект изображения соответствующего типа.

        Args:
            image_array: Массив с данными изображения
            breed: Порода животного
            url: URL изображения

        Returns:
            Объект ColorCatImage или GrayscaleCatImage
        """
        if len(image_array.shape) == 3:
                return ColorCatImage(image_array, breed, url)
        else:
            return GrayscaleCatImage(image_array.astype(np.uint8), breed, url)

    @timer_decorator
    def process_images(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Обработка всех загруженных изображений.

        Returns:
            Список кортежей с результатами обработки (пользовательский метод, библиотечный метод)
        """
        print("Начинаем обработку изображений...")

        if not self._downloaded_images:
            print("Нет изображений для обработки")
            return []

        results = []
        for i, cat_image in enumerate(self._downloaded_images):
            print(f"Обрабатываем изображение {i + 1} ({cat_image.breed})...")

            custom_edges = cat_image.detect_edges_custom()

            library_edges = cat_image.detect_edges_library()

            results.append((custom_edges, library_edges))
            print(f"Изображение {i + 1} обработано")

        return results

    @timer_decorator
    def save_images(self, output_dir: str = "processed_images") -> None:
        """
        Сохранение исходных и обработанных изображений.

        Args:
            output_dir: Директория для сохранения изображений
        """
        print(f"Сохранение изображений в директорию '{output_dir}'...")

        if not self._downloaded_images:
            print("Нет изображений для сохранения")
            return

        # Создаем директорию если не существует
        os.makedirs(output_dir, exist_ok=True)

        # Обрабатываем изображения
        processing_results = self.process_images()

        for i, (cat_image, (custom_edges, library_edges)) in enumerate(
                zip(self._downloaded_images, processing_results)
        ):
            # Генерируем безопасные имена файлов
            safe_breed = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in cat_image.breed)
            base_filename = f"{i + 1}_{safe_breed}"

            # Создаем пути с помощью os.path.join
            original_path = os.path.join(output_dir, f"{base_filename}_original.png")
            custom_path = os.path.join(output_dir, f"{base_filename}_custom_edges.png")
            library_path = os.path.join(output_dir, f"{base_filename}_library_edges.png")

            # Сохраняем исходное изображение
            try:
                original_image = Image.fromarray(cat_image.image_array)
                original_image.save(original_path)
            except Exception as e:
                print(f"Ошибка при сохранении оригинального изображения {i + 1}: {e}")
                continue

            # Сохраняем обработанные изображения
            try:
                custom_edges_image = Image.fromarray(custom_edges)
                custom_edges_image.save(custom_path)

                library_edges_image = Image.fromarray(library_edges)
                library_edges_image.save(library_path)

                print(f"Изображение {i + 1} сохранено:")
                print(f"Оригинал: {os.path.basename(original_path)}")
                print(f"Пользовательские контуры: {os.path.basename(custom_path)}")
                print(f"Библиотечные контуры: {os.path.basename(library_path)}")

            except Exception as e:
                print(f"Ошибка при сохранении обработанных изображений {i + 1}: {e}")

    def get_stats(self) -> dict:
        """
        Получение статистики по загруженным изображениям.

        Returns:
            Словарь со статистикой
        """
        stats = {
            'total_images': len(self._downloaded_images),
            'color_images': 0,
            'grayscale_images': 0,
            'breeds': set(),
            'shapes': []
        }

        for image in self._downloaded_images:
            stats['breeds'].add(image.breed)
            stats['shapes'].append(image.shape)

            if isinstance(image, ColorCatImage):
                stats['color_images'] += 1
            elif isinstance(image, GrayscaleCatImage):
                stats['grayscale_images'] += 1

        stats['breeds'] = list(stats['breeds'])
        return stats