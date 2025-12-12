import sys

print(f"Python path: {sys.path[:3]}...")

try:
    # Попробуем импортировать пакет
    import cat_processor

    print("✅ Пакет cat_processor успешно импортирован из другой директории!")

    # Проверим доступность компонентов
    from cat_processor import AsyncCatImageProcessor

    processor = AsyncCatImageProcessor()
    print(f"✅ Создан процессор: {processor}")

except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Убедитесь, что пакет установлен в текущем окружении:")
    print("pip list | grep cat-processor")
