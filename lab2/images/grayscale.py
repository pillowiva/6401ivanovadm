import numpy as np
import cv2
from .base import CatImage


class GrayscaleCatImage(CatImage):
    """
    Класс для работы с черно-белыми изображениями животных.

    Наследует от абстрактного класса CatImage и реализует
    специфичные для Ч/Б изображений методы обработки.

    Атрибуты:
        Наследует все атрибуты от CatImage
    """

    def detect_edges_custom(self) -> np.ndarray:
        return self._image_processing.edge_detection(self._image_array)

    def detect_edges_library(self) -> np.ndarray:
        laplacian = cv2.Laplacian(self._image_array, cv2.CV_64F)
        edges = np.uint8(np.absolute(laplacian))
        return edges

