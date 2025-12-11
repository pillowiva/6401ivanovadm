"""Модуль для работы с изображениями животных."""

from .base import CatImage
from .color import ColorCatImage
from .grayscale import GrayscaleCatImage

__all__ = ['CatImage', 'ColorCatImage', 'GrayscaleCatImage']