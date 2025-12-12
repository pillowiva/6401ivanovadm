import time
from functools import wraps
from typing import Any, Callable


def timer_decorator(func: Callable) -> Callable:
    """
    Декоратор для измерения времени выполнения методов.

    Args:
        func: Функция или метод, время выполнения которого нужно измерить

    Returns:
        Обернутая функция с измерением времени
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Метод {self.__class__.__name__}.{func.__name__} выполнен за {execution_time:.4f} секунд")
        return result

    return wrapper


def timer_decorator_static(func: Callable) -> Callable:
    """
    Декоратор для измерения времени выполнения статических методов.

    Args:
        func: Статический метод, время выполнения которого нужно измерить

    Returns:
        Обернутая функция с измерением времени
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Статический метод {func.__name__} выполнен за {execution_time:.4f} секунд")
        return result

    return wrapper