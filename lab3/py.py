import dill
import sys

def uses_fake_module():
    import broken_module  # Этот модуль НЕ существует
    return "Функция выполнена"

try:
    serialized = dill.dumps(uses_fake_module)
    print("Сериализация успешна")
    loaded_func = dill.loads(serialized)
    print(loaded_func())
except Exception as e:
    print(f"Ошибка: {type(e).__name__}")