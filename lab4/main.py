#!/usr/bin/env python3
"""
Лабораторная работа №4: Асинхронная обработка изображений животных

Основной файл программы. Содержит точку входа и пользовательский интерфейс.
"""

import sys
import os
import asyncio

# Добавляем пути для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Импортируем асинхронный процессор
from lab4.processor.image_processor import AsyncCatImageProcessor, async_main


def display_welcome() -> None:
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №4: АСИНХРОННАЯ ОБРАБОТКА ИЗОБРАЖЕНИЙ")
    print("=" * 60)
    print()
    print("Описание программы:")
    print("• Асинхронная загрузка изображений животных с The Cat API")
    print("• Параллельная обработка изображений в нескольких процессах")
    print("• Определение породы животного")
    print("• Выделение контуров пользовательскими и библиотечными методами")
    print("• Сохранение результатов в отдельную директорию")
    print("• Измерение времени выполнения операций")
    print()


def get_user_input() -> int:
    """
    Получение ввода от пользователя.

    Returns:
        Количество изображений для обработки

    Raises:
        SystemExit: При вводе 'q' или неверных данных
    """
    print("Введите количество изображений для обработки (1-10)")
    print("Или введите 'q' для выхода")

    while True:
        try:
            user_input = input(">>> ").strip().lower()

            if user_input == 'q':
                print("Выход из программы...")
                sys.exit(0)

            limit = int(user_input)

            if limit < 1:
                print("Количество изображений должно быть положительным числом")
                continue
            elif limit > 10:
                print("Рекомендуется не более 10 изображений для одного запуска")
                confirm = input("Продолжить? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue

            return limit

        except ValueError:
            print("Пожалуйста, введите целое число или 'q' для выхода")
        except KeyboardInterrupt:
            print("\nВыход из программы...")
            sys.exit(0)


def display_stats(processor: AsyncCatImageProcessor) -> None:
    """Отображение статистики по обработанным изображениям."""
    stats = processor.get_stats()

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
    display_welcome()

    # Проверяем API ключ напрямую
    processor = AsyncCatImageProcessor()
    if not processor._api_key:
        print("Ошибка: API_KEY не найден")
        return

    # Получаем ввод от пользователя
    limit = get_user_input()

    print(f"\nНачинаем обработку {limit} изображений...")

    # ЗАПУСКАЕМ асинхронную функцию напрямую!
    output_dir = "async_processed_images"

    # Вместо await используем asyncio.run()
    processor = asyncio.run(async_main(limit=limit, output_dir=output_dir))

    # Показываем статистику
    display_stats(processor)
    print("\nОБРАБОТКА ЗАВЕРШЕНА!")

if __name__ == "__main__":
    main()