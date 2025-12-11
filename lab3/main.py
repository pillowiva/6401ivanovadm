import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv_in_chunks(path, chunksize=100):
    """
    ГЕНЕРАТОР: Чтение CSV файла частями (чунками)
    Экономит память при работе с большими файлами
    Возвращает по одному DataFrame'у размера chunksize
    """
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield chunk

def extract_columns(chunks, columns, state_name=None):
    """
    ГЕНЕРАТОР: Фильтрация колонок и строк
    Из каждого чунка оставляет только нужные колонки
    Дополнительно фильтрует по штату если указан state_name
    """
    for chunk in chunks:
        df = chunk[columns]  # Оставляем только нужные колонки
        if state_name is not None:
            df = df[df['State'] == state_name]  # Фильтр по штату
        yield df

def moving_average(values, window=4):
    """
    Ручной расчет скользящего среднего
    values - список значений для сглаживания
    window - размер окна сглаживания
    """
    ma = []
    for i in range(len(values)):
        start = max(0, i - window + 1)  # Начало окна (не уходим в отрицательные индексы)
        ma.append(sum(values[start:i+1]) / (i - start + 1))  # Среднее в окне
    return ma

def aggregate_job_creation(chunks):
    cumulative_stats = pd.DataFrame(columns=['State', 'sum', 'count'])

    for df in chunks:
        # Группируем по штату и считаем сумму и количество
        chunk_stats = df.groupby('State')['Data.Calculated.Net Job Creation Rate'].agg(['sum', 'count']).reset_index()

        if cumulative_stats.empty:
            cumulative_stats = chunk_stats
        else:
            # Объединяем накопительную статистику с новой
            merged = pd.merge(cumulative_stats, chunk_stats, on='State', how='outer', suffixes=('_cum', '_chunk'))
            merged['sum'] = merged['sum_cum'].fillna(0) + merged['sum_chunk'].fillna(0)
            merged['count'] = merged['count_cum'].fillna(0) + merged['count_chunk'].fillna(0)
            cumulative_stats = merged[['State', 'sum', 'count']]

        yield cumulative_stats  # Возвращаем обновленную статистику

def aggregate_reallocation_variability(chunks):
    cumulative_stats = pd.DataFrame(columns=['State', 'sum', 'sum_sq', 'count'])

    for df in chunks:
        # Группируем и считаем три метрики: сумма, сумма квадратов, количество
        chunk_stats = df.groupby('State').agg({
            'Data.Calculated.Reallocation Rate': ['sum', lambda x: (x ** 2).sum(), 'count']
        }).reset_index()

        chunk_stats.columns = ['State', 'sum', 'sum_sq', 'count']

        if cumulative_stats.empty:
            cumulative_stats = chunk_stats
        else:
            # Объединяем статистику
            merged = pd.merge(cumulative_stats, chunk_stats, on='State', how='outer', suffixes=('_cum', '_chunk'))
            merged['sum'] = merged['sum_cum'].fillna(0) + merged['sum_chunk'].fillna(0)
            merged['sum_sq'] = merged['sum_sq_cum'].fillna(0) + merged['sum_sq_chunk'].fillna(0)
            merged['count'] = merged['count_cum'].fillna(0) + merged['count_chunk'].fillna(0)
            cumulative_stats = merged[['State', 'sum', 'sum_sq', 'count']]

        yield cumulative_stats

def aggregate_job_destruction_by_year(chunks):
    cumulative_stats = pd.DataFrame(columns=['Year', 'sum', 'count'])

    for df in chunks:
        chunk_stats = df.groupby('Year')['Data.Job Destruction.Rate'].agg(['sum', 'count']).reset_index()

        if cumulative_stats.empty:
            cumulative_stats = chunk_stats
        else:
            merged = pd.merge(cumulative_stats, chunk_stats, on='Year', how='outer', suffixes=('_cum', '_chunk'))
            merged['sum'] = merged['sum_cum'].fillna(0) + merged['sum_chunk'].fillna(0)
            merged['count'] = merged['count_cum'].fillna(0) + merged['count_chunk'].fillna(0)
            cumulative_stats = merged[['Year', 'sum', 'count']]

        yield cumulative_stats

def finalize_averages(stats_stream):
    final_stats = None
    for stats in stats_stream:
        final_stats = stats

    if final_stats is None or final_stats.empty:
        return pd.Series(dtype=float)

    final_stats['average'] = final_stats['sum'] / final_stats['count']  # Вычисляем среднее
    return final_stats.set_index('State')['average']

def finalize_variability(stats_stream):
    final_stats = None
    for stats in stats_stream:
        final_stats = stats

    if final_stats is None or final_stats.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    final_stats['mean'] = final_stats['sum'] / final_stats['count']
    final_stats['variance'] = (final_stats['sum_sq'] / final_stats['count']) - (final_stats['mean'] ** 2)
    final_stats['std'] = np.sqrt(final_stats['variance'])

    return (final_stats.set_index('State')['std'],
            final_stats.set_index('State')['mean'],
            final_stats.set_index('State')['count'])

def finalize_by_year(stats_stream):
    final_stats = None
    for stats in stats_stream:
        final_stats = stats

    if final_stats is None or final_stats.empty:
        return pd.Series(dtype=float)

    final_stats['average'] = final_stats['sum'] / final_stats['count']
    return final_stats.set_index('Year')['average']

def plot_task1(data, top=3, bottom=3):
    sorted_data = data.sort_values()  # Сортируем по возрастанию
    lowest = sorted_data.head(bottom)  # bottom самых низких значений
    highest = sorted_data.tail(top)    # top самых высоких значений
    selected = pd.concat([lowest, highest])  # Объединяем

    labels = selected.index
    values = selected.values

    plt.figure(figsize=(11, 5))
    plt.bar(labels, values)
    plt.title("Средние темпы создания рабочих мест по штатам")
    plt.ylabel("Средний рост занятости")
    plt.tight_layout()
    plt.show()

def plot_task2(std_data, mean_data, count_data, top=3, bottom=3):
    sorted_std = std_data.sort_values()  # Сортируем по стандартному отклонению
    most_stable = sorted_std.head(bottom)   # Самые стабильные (низкое std)
    most_turbulent = sorted_std.tail(top)   # Самые турбулентные (высокое std)

    labels = most_stable.index.tolist() + most_turbulent.index.tolist()
    std_values = most_stable.values.tolist() + most_turbulent.values.tolist()
    mean_values = [mean_data[label] for label in labels]  # Средние значения для каждого штата
    n_values = [count_data[label] for label in labels]    # Количество наблюдений

    se = []  # Standard Error
    ci = []  # Confidence Interval
    for std, n in zip(std_values, n_values):
        se_value = std / np.sqrt(n) if n > 1 else 0  # Стандартная ошибка
        se.append(se_value)
        ci.append(1.96 * se_value)  # 95% доверительный интервал

    plt.figure(figsize=(11, 5))
    plt.bar(labels, mean_values, yerr=ci, capsize=6)  # yerr - линии погрешностей
    plt.title("Средние показатели трудоустройства по штатам")
    plt.ylabel("Темп трудоустройства")
    plt.tight_layout()
    plt.show()

def plot_task3(state_name, mean_by_year, window=4):
    years = mean_by_year.index.tolist()
    values = mean_by_year.values.tolist()
    ma_values = moving_average(values, window=window)  # Вычисляем скользящее среднее

    plt.figure(figsize=(12, 5))
    plt.plot(years, values, label='Фактическое сокращение')
    plt.plot(years, ma_values, linestyle='--', label='Скользящее среднее')
    plt.title(f'Скорость сокращения рабочих мест в штате {state_name}')
    plt.ylabel('Скорость сокращения рабочих мест')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_pipeline_task1(csv_path):
    p1 = read_csv_in_chunks(csv_path, chunksize=500)  # Чтение чунками
    p2 = extract_columns(p1, ['State', 'Data.Calculated.Net Job Creation Rate'])  # Фильтрация колонок
    p3 = aggregate_job_creation(p2)  # Агрегация данных
    result = finalize_averages(p3)   # Вычисление средних
    plot_task1(result)               # Визуализация

def run_pipeline_task2(csv_path):
    p1 = read_csv_in_chunks(csv_path, chunksize=500)
    p2 = extract_columns(p1, ['State', 'Data.Calculated.Reallocation Rate'])
    p3 = aggregate_reallocation_variability(p2)
    std_data, mean_data, count_data = finalize_variability(p3)
    plot_task2(std_data, mean_data, count_data)
    return std_data  # Возвращаем std для использования в задаче 3

def run_pipeline_task3(csv_path, std_data, window=4):
    unstable_state = std_data.idxmax()  # Штат с максимальным стандартным отклонением
    p1 = read_csv_in_chunks(csv_path)
    p2 = extract_columns(p1, ['State', 'Year', 'Data.Job Destruction.Rate'], state_name=unstable_state)
    p3 = aggregate_job_destruction_by_year(p2)
    result_by_year = finalize_by_year(p3)
    plot_task3(unstable_state, result_by_year, window=window)

if __name__ == "__main__":
    run_pipeline_task1("business_dynamics.csv")  # Задача 1: 3 штата с наибольшим и 3 с наименьшим средним темпом создания рабочих мест (Net Job Creation Rate)
    std_data = run_pipeline_task2("business_dynamics.csv")  # Задача 2: 3 штата с наиболее стабильным рынком труда и 3 с наиболее турбулентным – по величине разброса показателя (Reallocation Rate)
    run_pipeline_task3("business_dynamics.csv", std_data)  # Задача 3: Динамика темпа закрытия рабочих мест (Job Destruction Rate) для наиболее нестабильного штата за все время наблюдений