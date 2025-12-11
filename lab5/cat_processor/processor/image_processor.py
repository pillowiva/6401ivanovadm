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

from utils.timer import timer_decorator
from cat_processor import ColorCatImage, GrayscaleCatImage
from utils import get_module_logger

# Загрузка переменных окружения
load_dotenv()

# Логгер модуля (основной)
logger = get_module_logger("processor")


def process_single_image(image_data: dict) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Обрабатывает одно изображение в отдельном процессе.

    Args:
        image_data: Данные изображения

    Returns:
        Кортеж (base_filename, custom_edges, library_edges)
    """
    # Отдельный логгер для подпроцесса
    process_logger = get_module_logger("processor.process")

    index = image_data["index"]
    breed = image_data["breed"]
    image_array = image_data["image_array"]
    url = image_data["url"]

    # Кратко — в консоль и файл
    process_logger.info("Начало обработки изображения %s", index)

    # Подробно — только в файл
    process_logger.debug(
        "Начало свёртки для изображения %s "
        "(pid=%s, breed=%s, shape=%s, url=%s)",
        index,
        os.getpid(),
        breed,
        image_array.shape,
        url,
    )

    # Создаём объект изображения
    if len(image_array.shape) == 3:
        process_logger.debug("Изображение %s распознано как цветное", index)
        cat_image = ColorCatImage(image_array, breed, url)
    else:
        process_logger.debug("Изображение %s распознано как Ч/Б", index)
        cat_image = GrayscaleCatImage(image_array.astype(np.uint8), breed, url)

    process_logger.debug("Запуск пользовательского алгоритма выделения контуров для %s", index)
    custom_edges = cat_image.detect_edges_custom()

    process_logger.debug("Запуск библиотечного алгоритма выделения контуров для %s", index)
    library_edges = cat_image.detect_edges_library()

    # Генерируем безопасное имя файла
    safe_breed = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in breed)
    base_filename = f"{index}_{safe_breed}"

    process_logger.debug(
        "Свёртка для изображения %s завершена (pid=%s, base_filename=%s)",
        index,
        os.getpid(),
        base_filename,
    )
    process_logger.info("Обработка изображения %s завершена", index)

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

        Args:
            api_url: URL API для получения изображений
        """
        self._api_url = api_url
        self._api_key = os.getenv("API_KEY")
        self._image_data: List[dict] = []

        logger.info("Создан AsyncCatImageProcessor")
        logger.debug(
            "Инициализация AsyncCatImageProcessor: api_url=%s, api_key_present=%s",
            api_url,
            bool(self._api_key),
        )

        if not self._api_key:
            logger.warning("API_KEY не найден в переменных окружения")

    @property
    def image_data(self) -> List[dict]:
        """Property для получения данных изображений."""
        return self._image_data

    async def download_single_image(
        self,
        session: aiohttp.ClientSession,
        item,
        index: int,
        output_dir: str,
    ) -> None:
        """
        Асинхронно скачивает и сохраняет одно изображение.

        Args:
            session: aiohttp сессия
            item: Данные изображения из API
            index: Порядковый номер изображения
            output_dir: Директория для сохранения
        """
        logger.info("Начало загрузки изображения %s", index)
        logger.debug("Downloading image %s started. Raw item: %s", index, item)

        try:
            # Логируем структуру данных
            logger.debug(
                "Изображение %s — ключи данных: %s",
                index,
                list(item.keys()),
            )

            if "breeds" in item:
                logger.debug(
                    "Изображение %s — информация о породах: %s",
                    index,
                    item["breeds"],
                )

            # Скачиваем изображение асинхронно
            async with session.get(item["url"]) as response:
                if response.status == 200:
                    image_content = await response.read()
                    logger.debug(
                        "Изображение %s успешно загружено, размер=%d байт",
                        index,
                        len(image_content),
                    )

                    # Конвертация в numpy array
                    image = Image.open(io.BytesIO(image_content))
                    image_array = np.array(image)
                    logger.debug(
                        "Изображение %s — numpy shape=%s, dtype=%s",
                        index,
                        image_array.shape,
                        image_array.dtype,
                    )

                    breed_name = "unknown"
                    if item.get("breeds"):
                        breed_info = item["breeds"][0]
                        breed_name = (
                            breed_info.get("name", "unknown").replace(" ", "_").lower()
                        )
                        logger.debug(
                            "Изображение %s — определена порода: %s",
                            index,
                            breed_name,
                        )

                    # Генерируем безопасное имя файла
                    safe_breed = "".join(
                        c if c.isalnum() or c in ("-", "_") else "_" for c in breed_name
                    )
                    original_filename = f"{index}_{safe_breed}_original.png"
                    original_path = os.path.join(output_dir, original_filename)

                    # Сохраняем оригинальное изображение асинхронно
                    async with aiofiles.open(original_path, "wb") as f:
                        await f.write(image_content)

                    logger.debug(
                        "Изображение %s сохранено по пути: %s",
                        index,
                        original_path,
                    )

                    # Сохраняем данные для последующей обработки
                    self._image_data.append(
                        {
                            "index": index,
                            "breed": breed_name,
                            "url": item["url"],
                            "image_array": image_array,
                            "original_path": original_path,
                        }
                    )

                    logger.info("Загрузка изображения %s завершена", index)
                else:
                    error_msg = f"Ошибка загрузки изображения {index}: HTTP {response.status}"
                    logger.error(error_msg)
                    logger.info(
                        "Загрузка изображения %s завершилась с ошибкой HTTP %s",
                        index,
                        response.status,
                    )

        except Exception as e:
            error_msg = f"Ошибка загрузки изображения {index}: {e}"
            logger.error(error_msg, exc_info=True)
            logger.info("Ошибка при загрузке изображения %s: %s", index, e)

    @timer_decorator
    async def download_images_async(
        self,
        limit: int = 1,
        output_dir: str = "processed_images",
    ) -> None:
        """
        Асинхронно загружает изображения с API.
        """
        logger.info("Старт асинхронной загрузки %s изображений", limit)
        logger.debug(
            "Параметры асинхронной загрузки: limit=%s, output_dir=%s, api_url=%s",
            limit,
            output_dir,
            self._api_url,
        )

        if not self._api_key:
            error_msg = "API_KEY не найден в переменных окружения"
            logger.error(error_msg)
            logger.info("Загрузка прервана: отсутствует API_KEY")
            return

        # Создаем директорию
        os.makedirs(output_dir, exist_ok=True)
        logger.debug("Создана/проверена директория: %s", output_dir)

        params = {
            "limit": limit,
            "api_key": self._api_key,
            "has_breeds": 1,
        }
        logger.debug("Параметры запроса к API: %s", params)

        try:
            # NB: отключаем проверку SSL (учебный костыль)
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                logger.debug("Отправка запроса к API: %s", self._api_url)
                async with session.get(self._api_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                logger.info("Получено %s изображений от API", len(data))
                logger.debug("Сырые данные от API: %s", data)

                # Создаём задачи для скачивания
                tasks = []
                for i, item in enumerate(data):
                    task = self.download_single_image(
                        session,
                        item,
                        i + 1,
                        output_dir,
                    )
                    tasks.append(task)
                    logger.debug(
                        "Создана задача загрузки для изображения %s", i + 1
                    )

                logger.debug(
                    "Запуск %s задач загрузки параллельно", len(tasks)
                )
                await asyncio.gather(*tasks)

                logger.info(
                    "Успешно загружено %s изображений", len(self._image_data)
                )

        except aiohttp.ClientError as e:
            error_msg = f"Ошибка сети при загрузке изображений: {e}"
            logger.error(error_msg, exc_info=True)
            logger.info("Сетевая ошибка при загрузке изображений: %s", e)
            raise
        except Exception as e:
            error_msg = f"Ошибка во время асинхронной загрузки: {e}"
            logger.error(error_msg, exc_info=True)
            logger.info("Общая ошибка асинхронной загрузки: %s", e)
            raise

    @timer_decorator
    def process_images_parallel(self, output_dir: str = "processed_images") -> None:
        """
        Обрабатывает изображения параллельно в нескольких процессах.

        Args:
            output_dir: Директория для сохранения обработанных изображений
        """
        logger.info("Начало параллельной обработки изображений")

        if not self._image_data:
            logger.warning("Нет изображений для обработки")
            return

        logger.info("Количество изображений для обработки: %s", len(self._image_data))

        # Сортируем данные по индексу для сохранения порядка
        sorted_data = sorted(self._image_data, key=lambda x: x["index"])
        logger.debug("Изображения отсортированы по индексу: %s", [d["index"] for d in sorted_data])

        # Используем ProcessPoolExecutor для параллельной обработки
        with ProcessPoolExecutor() as executor:
            logger.debug("Создан ProcessPoolExecutor для параллельной обработки")

            futures = [
                executor.submit(process_single_image, data) for data in sorted_data
            ]
            logger.debug("Создано %s задач обработки", len(futures))

            results: List[Tuple] = []
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info("Обработка изображения %s завершена", i + 1)
                except Exception as e:
                    error_msg = f"Ошибка обработки изображения {i + 1}: {e}"
                    logger.error(error_msg, exc_info=True)

        # Сохраняем обработанные изображения
        self._save_processed_images(results, output_dir)

        logger.info("Параллельная обработка изображений завершена")

    def _save_processed_images(self, results: List[Tuple], output_dir: str) -> None:
        """
        Сохраняет обработанные изображения.

        Args:
            results: Результаты обработки
            output_dir: Директория для сохранения
        """
        logger.info("Сохранение обработанных изображений...")
        logger.debug("Выходной каталог для сохранения результатов: %s", output_dir)

        for base_filename, custom_edges, library_edges in results:
            try:
                custom_path = os.path.join(
                    output_dir, f"{base_filename}_custom_edges.png"
                )
                library_path = os.path.join(
                    output_dir, f"{base_filename}_library_edges.png"
                )

                Image.fromarray(custom_edges).save(custom_path)
                Image.fromarray(library_edges).save(library_path)

                logger.info("Обработанные изображения сохранены: %s", base_filename)
                logger.debug(
                    "Пути сохранения для %s: custom=%s, library=%s",
                    base_filename,
                    custom_path,
                    library_path,
                )

            except Exception as e:
                error_msg = (
                    f"Ошибка сохранения обработанных изображений для "
                    f"{base_filename}: {e}"
                )
                logger.error(error_msg, exc_info=True)

    def get_stats(self) -> dict:
        """
        Получение статистики по загруженным изображениям.

        Returns:
            Словарь со статистикой
        """
        logger.debug("Формирование статистики по обработанным изображениям")

        stats = {
            "total_images": len(self._image_data),
            "color_images": 0,
            "grayscale_images": 0,
            "breeds": set(),
            "shapes": [],
        }

        for data in self._image_data:
            stats["breeds"].add(data["breed"])
            stats["shapes"].append(data["image_array"].shape)

            if len(data["image_array"].shape) == 3:
                stats["color_images"] += 1
            else:
                stats["grayscale_images"] += 1

        stats["breeds"] = list(stats["breeds"])
        logger.debug("Статистика сформирована: %s", stats)
        return stats


async def async_main(
    limit: int = 5,
    output_dir: str = "async_processed_images",
):
    """
    Основная асинхронная функция обработки изображений.
    """
    logger.info("=== Асинхронный процессор изображений кошек ===")
    logger.info("Запуск async_main: limit=%s, output_dir=%s", limit, output_dir)

    processor = AsyncCatImageProcessor()
    logger.debug("Создан экземпляр AsyncCatImageProcessor в async_main")

    try:
        start_time = time.time()
        logger.debug("Начало обработки, время старта: %s", start_time)

        await processor.download_images_async(limit=limit, output_dir=output_dir)
        processor.process_images_parallel(output_dir=output_dir)

        stats = processor.get_stats()

        logger.info("=== Статистика ===")
        logger.info("Всего изображений: %s", stats["total_images"])
        logger.info("Цветных изображений: %s", stats["color_images"])
        logger.info("Ч/Б изображений: %s", stats["grayscale_images"])
        logger.info("Породы: %s", stats["breeds"])

        end_time = time.time()
        total_time = end_time - start_time
        logger.info("Общее время выполнения: %.2f секунд", total_time)
        logger.debug(
            "Завершение async_main: start=%s, end=%s, total=%.4f",
            start_time,
            end_time,
            total_time,
        )

        return processor

    except Exception as e:
        error_msg = f"Выполнение программы завершилось с ошибкой: {e}"
        logger.error(error_msg, exc_info=True)
        raise
