#!/usr/bin/env python3
"""
Скрипт для запуска всех тестов проекта.
"""

import unittest
import sys
import os


def run_all_tests():
    """Запуск всех тестов."""
    # Получаем абсолютный путь к корневой директории проекта (lab5)
    project_root = os.path.abspath(os.path.dirname(__file__))

    # Добавляем корень проекта в sys.path
    sys.path.insert(0, project_root)
    # Получаем абсолютный путь к директории тестов
    tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests')


    # Проверяем, существует ли директория tests
    if not os.path.exists(tests_dir):
        print(f"Ошибка: Директория tests не найдена: {tests_dir}")
        return 1

    # Добавляем корень проекта в sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    # Находим все тесты
    test_loader = unittest.TestLoader()

    try:
        test_suite = test_loader.discover(
            start_dir=tests_dir,
            pattern='test_*.py',
            top_level_dir=project_root
        )

        # Запускаем тесты
        test_runner = unittest.TextTestRunner(verbosity=2)
        result = test_runner.run(test_suite)

        # Возвращаем код выхода
        return 0 if result.wasSuccessful() else 1

    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print(f"\nПуть поиска модулей (sys.path):")
        for p in sys.path:
            print(f"  {p}")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)