import numpy as np
import cv2
from .base import CatImage

class ColorCatImage(CatImage):
    """
    Класс для работы с цветными изображениями животных.

    Наследует от абстрактного класса CatImage и реализует
    специфичные для цветных изображений методы обработки.

    Атрибуты:
        Наследует все атрибуты от CatImage
    """

    def detect_edges_custom(self) -> np.ndarray:
        return self._image_processing.edge_detection(self._image_array)

    def detect_edges_library(self) -> np.ndarray:
        if len(self._image_array.shape) == 3:
            gray = cv2.cvtColor(self._image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = self._image_array

        edges = cv2.Canny(gray, 50, 150)
        return edges