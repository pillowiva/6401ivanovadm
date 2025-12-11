from abc import ABC, abstractmethod
import numpy as np
from lab1.implementation.image_processing import ImageProcessing
import cv2

class CatImage(ABC):
    """
    Абстрактный базовый класс для изображений животных.

    Этот класс определяет общий интерфейс для всех типов изображений
    и реализует общую функциональность (перегрузка операторов, свойства).

    Атрибуты:
        _image_array (np.ndarray): Массив с данными изображения
        _breed (str): Порода животного
        _image_url (str): URL исходного изображения
    """

    def __init__(self, image_array: np.ndarray, breed: str, image_url: str) -> None:
        self._image_array = image_array
        self._breed = breed
        self._image_url = image_url
        self._image_processing = ImageProcessing()

    @property
    def image_array(self) -> np.ndarray:
        """Property для получения массива изображения."""
        return self._image_array

    @property
    def breed(self) -> str:
        """Property для получения породы животного."""
        return self._breed

    @property
    def image_url(self) -> str:
        """Property для получения URL изображения."""
        return self._image_url

    @property
    def shape(self) -> tuple:
        """Property для получения формы массива изображения."""
        return self._image_array.shape

    @abstractmethod
    def detect_edges_custom(self) -> np.ndarray:
        pass

    @abstractmethod
    def detect_edges_library(self) -> np.ndarray:
        pass

    def _prepare_images(self, other: 'CatImage') -> tuple:
        """
        Приводит два изображения к одинаковому размеру и количеству каналов.
        """
        img1 = self.image_array
        img2 = other.image_array

        # Приводим к одинаковому количеству каналов
        if len(img1.shape) != len(img2.shape):
            if len(img1.shape) == 2:
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            else:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        # Приводим к одинаковому размеру (берем минимальные размеры)
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])

        # Обрезаем оба изображения до общего размера по центру
        def crop_center(img, h, w):
            y = (img.shape[0] - h) // 2
            x = (img.shape[1] - w) // 2
            return img[y:y + h, x:x + w]

        return crop_center(img1, h, w), crop_center(img2, h, w)

    def __add__(self, other: 'CatImage') -> 'CatImage':
        """
        Перегрузка оператора сложения для изображений.
        """
        img1, img2 = self._prepare_images(other)
        result_array = np.clip(img1.astype(int) + img2.astype(int), 0, 255).astype(np.uint8)
        return self.__class__(result_array, f"{self.breed}_added", self.image_url)

    def __sub__(self, other: 'CatImage') -> 'CatImage':
        """
        Перегрузка оператора вычитания для изображений.
        """
        img1, img2 = self._prepare_images(other)
        result_array = np.clip(img1.astype(int) - img2.astype(int), 0, 255).astype(np.uint8)
        return self.__class__(result_array, f"{self.breed}_subtracted", self.image_url)

    def __str__(self) -> str:
        """
        Перегрузка метода преобразования в строку.

        Returns:
            Строковое представление объекта
        """
        return (f"CatImage(breed='{self.breed}', shape={self.image_array.shape}, "
                f"dtype={self.image_array.dtype}, url='{self.image_url}')")

    def __repr__(self) -> str:
        """
        Репрезентация объекта для отладки.

        Returns:
            Строковое представление объекта для отладки
        """
        return self.__str__()