#!/usr/bin/env python3
"""
Лабораторная работа №4: Асинхронная обработка изображений животных

Основной файл программы. Содержит точку входа и пользовательский интерфейс.
"""

import sys
import os
import asyncio
from pathlib import Path

# Добавляем пути для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Импортируем конфигурацию логирования
from utils import get_module_logger

# Импортируем асинхронный процессор
from cat_processor import AsyncCatImageProcessor, async_main

# Создаем логгер для main.py
logger = get_module_logger("main")

def display_welcome() -> None:
    """Вывод приветственного сообщения."""
    welcome_msg = """im
    ============================================================
    ЛАБОРАТОРНАЯ РАБОТА №4: АСИНХРОННАЯ ОБРАБОТКА ИЗОБРАЖЕНИЙ
    ============================================================

    Описание программы:
    • Асинхронная загрузка изображений животных с The Cat API
    • Параллельная обработка изображений в нескольких процессах
    • Определение породы животного
    • Выделение контуров пользовательскими и библиотечными методами
    • Сохранение результатов в отдельную директорию
    • Измерение времени выполнения операций
    """

    logger.info("=" * 60)
    logger.info("ЛАБОРАТОРНАЯ РАБОТА №4: АСИНХРОННАЯ ОБРАБОТКА ИЗОБРАЖЕНИЙ")
    logger.info("=" * 60)
    logger.info("\nОписание программы:")
    logger.info("• Асинхронная загрузка изображений животных с The Cat API")
    logger.info("• Параллельная обработка изображений в нескольких процессах")
    logger.info("• Определение породы животного")
    logger.info("• Выделение контуров пользовательскими и библиотечными методами")
    logger.info("• Сохранение результатов в отдельную директорию")
    logger.info("• Измерение времени выполнения операций")

    print(welcome_msg)

def get_user_input() -> int:
    """
    Получение ввода от пользователя.

    Returns:
        Количество изображений для обработки

    Raises:
        SystemExit: При вводе 'q' или неверных данных
    """
    logger.info("Запрос ввода от пользователя")

    print("Введите количество изображений для обработки (1-10)")
    print("Или введите 'q' для выхода")

    while True:
        try:
            user_input = input(">>> ").strip().lower()

            if user_input == 'q':
                logger.info("Пользователь выбрал выход из программы")
                print("Выход из программы...")
                sys.exit(0)

            limit = int(user_input)

            if limit < 1:
                logger.warning(f"Введено некорректное значение: {limit} (должно быть положительным)")
                print("Количество изображений должно быть положительным числом")
                continue
            elif limit > 10:
                logger.warning(f"Введено большое количество изображений: {limit}")
                print("Рекомендуется не более 10 изображений для одного запуска")
                confirm = input("Продолжить? (y/n): ").strip().lower()
                if confirm != 'y':
                    logger.info("Пользователь отменил обработку большого количества изображений")
                    continue
                else:
                    logger.info("Пользователь подтвердил обработку большого количества изображений")

            logger.info(f"Пользователь выбрал обработку {limit} изображений")
            return limit

        except ValueError:
            logger.warning("Введено некорректное значение (не число)")
            print("Пожалуйста, введите целое число или 'q' для выхода")
        except KeyboardInterrupt:
            logger.info("Программа прервана пользователем (Ctrl+C)")
            print("\nВыход из программы...")
            sys.exit(0)

def display_stats(processor: AsyncCatImageProcessor) -> None:
    """Отображение статистики по обработанным изображениям."""
    stats = processor.get_stats()

    logger.info("=" * 40)
    logger.info("СТАТИСТИКА ОБРАБОТКИ")
    logger.info("=" * 40)
    logger.info(f"Всего изображений: {stats['total_images']}")
    logger.info(f"Цветных изображений: {stats['color_images']}")
    logger.info(f"Ч/Б изображений: {stats['grayscale_images']}")
    logger.info(f"Уникальных пород: {len(stats['breeds'])}")

    if stats['breeds']:
        logger.info("Список пород:")
        for breed in stats['breeds']:
            logger.info(f"   • {breed}")

    logger.info("=" * 40)

    print("\n" + "=" * 40)
    print("СТАТИСТИКА ОБРАБОТКИ")
    print("=" * 40)
    print(f"Всего изображений: {stats['total_images']}")
    print(f"Цветных изображений: {stats['color_images']}")
    print(f"Ч/Б изображений: {stats['grayscale_images']}")
    print(f"Уникальных пород: {len(stats['breeds'])}")

    if stats['breeds']:
        print("Список пород:")
        for breed in stats['breeds']:
            print(f"   • {breed}")

    print("=" * 40)

def main() -> None:
    """Точка входа в программу - УПРОЩЕННАЯ ВЕРСИЯ"""
    logger.info("=" * 60)
    logger.info("Запуск программы обработки изображений")
    logger.info("=" * 60)

    display_welcome()

    # Проверяем API ключ напрямую
    processor = AsyncCatImageProcessor()
    if not processor._api_key:
        logger.error("API_KEY не найден в переменных окружения")
        print("Ошибка: API_KEY не найден")
        return
    else:
        logger.info("API_KEY успешно загружен")

    # Получаем ввод от пользователя
    limit = get_user_input()

    logger.info(f"Начинаем обработку {limit} изображений...")
    print(f"\nНачинаем обработку {limit} изображений...")

    # ЗАПУСКАЕМ асинхронную функцию напрямую!
    output_dir = "async_processed_images"

    logger.info(f"Создание директории для результатов: {output_dir}")
    Path(output_dir).mkdir(exist_ok=True)

    try:
        # Вместо await используем asyncio.run()
        logger.debug(f"Запуск асинхронной обработки с параметрами: limit={limit}, output_dir={output_dir}")
        processor = asyncio.run(async_main(limit=limit, output_dir=output_dir))
        logger.info("Асинхронная обработка успешно завершена")

        # Показываем статистику
        display_stats(processor)
        logger.info("ОБРАБОТКА ЗАВЕРШЕНА!")
        print("\nОБРАБОТКА ЗАВЕРШЕНА!")

    except Exception as e:
        logger.error(f"Ошибка во время обработки: {e}", exc_info=True)
        print(f"\nПроизошла ошибка во время обработки: {e}")
    finally:
        logger.info("Программа завершена")

if __name__ == "__main__":
    main()