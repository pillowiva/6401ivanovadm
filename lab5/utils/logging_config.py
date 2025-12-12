"""
Конфигурация логирования для приложения обработки изображений.

Настраивает два обработчика:
1. Файловый (уровень DEBUG) - подробные логи с информацией о файле и строке
2. Консольный (уровень INFO) - краткие логи для пользователя
"""

import logging
import sys
from pathlib import Path

# Создаем директорию для логов, если её нет
BASE_DIR = Path(__file__).resolve().parent.parent
log_dir = BASE_DIR / "logs"
log_dir.mkdir(exist_ok=True)

def setup_logging():
    """
    Настройка конфигурации логирования.
    """
    # Создаем корневой логгер
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Удаляем существующие обработчики, чтобы избежать дублирования
    logger.handlers.clear()

    # 1. Файловый обработчик (уровень DEBUG)
    file_handler = logging.FileHandler(
        filename=log_dir / "app.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    # Подробный формат для файла
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # 2. Консольный обработчик (уровень INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Краткий формат для консоли
    console_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # Добавляем обработчики к логгеру
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Логируем сам факт настройки логирования
    logger.debug("Логирование настроено: файл %s и консоль", log_dir / "app.log")

    return logger

# Инициализируем логирование ПРИ ИМПОРТЕ МОДУЛЯ
setup_logging()

# Создаем именованные логгеры для разных модулей
def get_module_logger(module_name):
    """
    Возвращает настроенный логгер для конкретного модуля.

    Args:
        module_name: Имя модуля для логирования

    Returns:
        Настроенный логгер
    """
    return logging.getLogger(module_name)

# Создаем стандартные логгеры для разных частей приложения
main_logger = get_module_logger("main")
processor_logger = get_module_logger("processor")