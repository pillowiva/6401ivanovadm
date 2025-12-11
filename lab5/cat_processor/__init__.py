"""Пакет для обработки изображений животных."""

from .images.color import ColorCatImage
from .images.grayscale import GrayscaleCatImage
from .images.base import CatImage
from .processor.image_processor import AsyncCatImageProcessor, async_main, process_single_image

__version__ = "1.0.0"
__author__ = ""
__all__ = [
    'CatImage',
    'ColorCatImage',
    'GrayscaleCatImage',
    'AsyncCatImageProcessor',
    'async_main',
    'process_single_image'
]