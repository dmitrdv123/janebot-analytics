import pandas as pd
import matplotlib.pyplot as plt
import math
import cmath
import random
import numpy as np

def sigmoid(x):
    """Сигмоидная функция для нормализации."""
    return 1 / (1 + math.exp(-x))

def compute_hamiltonian(df, current_idx, lookback=5):
    """Рассчитывает гамильтониан на основе рыночных данных."""
    row = df.iloc[current_idx]
    range_price = row["highPrice"] - row["lowPrice"]
    volume = row["volume"]
    turnover = row["turnover"]

    # Исторические данные
    start_idx = max(0, current_idx - lookback)
    hist_df = df.iloc[start_idx:current_idx + 1]
    if not hist_df.empty:
        avg_range = (hist_df["highPrice"] - hist_df["lowPrice"]).mean()
        avg_volume = hist_df["volume"].mean()
        avg_turnover = hist_df["turnover"].mean()
    else:
        avg_range = range_price
        avg_volume = volume
        avg_turnover = turnover

    # Гамильтониан: энергия системы
    h = (range_price / avg_range) * (volume / avg_volume)  # Волатильность и активность
    h += turnover / avg_turnover  # Влияние денежного потока
    return h / 1000  # Нормализация для численной стабильности

def evolve_wave_function(prev_amps, h, dt=1):
    """Эволюция волновой функции с уравнением Шрёдингера."""
    growth_amp, decline_amp, stagnation_amp = prev_amps
    # Унитарный оператор: U = exp(-iHt/ħ), где ħ = 1 для упрощения
    u = cmath.exp(-1j * h * dt)
    
    # Эволюция амплитуд
    new_growth_amp = growth_amp * u
    new_decline_amp = decline_amp * u
    new_stagnation_amp = stagnation_amp * u
    
    # Нормализация
    norm = math.sqrt(abs(new_growth_amp)**2 + abs(new_decline_amp)**2 + abs(new_stagnation_amp)**2)
    if norm > 0:
        new_growth_amp /= norm
        new_decline_amp /= norm
        new_stagnation_amp /= norm
    
    return new_growth_amp, new_decline_amp, new_stagnation_amp

def calculate_wave_function(df, current_idx, prev_amps=None, lookback=5):
    """Рассчитывает волновую функцию с запутанностью и эволюцией."""
    row = df.iloc[current_idx]
    open_price = row["openPrice"]
    high_price = row["highPrice"]
    low_price = row["lowPrice"]
    close_price = row["closePrice"]
    volume = row["volume"]
    turnover = row["turnover"]

    # Анализ текущей свечи
    price_change = close_price - open_price
    range_price = high_price - low_price
    range_percent = range_price / open_price * 100

    # Исторические данные
    start_idx = max(0, current_idx - lookback)
    hist_df = df.iloc[start_idx:current_idx]
    
    if not hist_df.empty:
        hist_price_changes = hist_df["closePrice"] - hist_df["openPrice"]
        avg_price_change = hist_price_changes.mean()
        hist_ranges = hist_df["highPrice"] - hist_df["lowPrice"]
        avg_range = hist_ranges.mean()
        avg_volume = hist_df["volume"].mean()
        trend_strength = np.corrcoef(hist_df["startTime"], hist_df["closePrice"])[0, 1] if len(hist_df) > 1 else 0
    else:
        avg_price_change = price_change
        avg_range = range_price
        avg_volume = volume
        trend_strength = 0

    # Начальные амплитуды (если нет предыдущих)
    if prev_amps is None:
        growth_amp = decline_amp = stagnation_amp = complex(1 / math.sqrt(3), 0)
    else:
        # Эволюция от предыдущего состояния
        h = compute_hamiltonian(df, current_idx - 1, lookback)
        growth_amp, decline_amp, stagnation_amp = evolve_wave_function(prev_amps, h)

    # Корректировка амплитуд текущей свечой
    if price_change > 0:
        growth_amp += 0.4 if close_price == high_price else 0.3
        decline_amp += 0.1
        stagnation_amp += 0.2
    elif price_change < 0:
        decline_amp += 0.4 if close_price == low_price else 0.3
        growth_amp += 0.1
        stagnation_amp += 0.2
    else:
        stagnation_amp += 0.5
        growth_amp += 0.25
        decline_amp += 0.25

    # Интерференция с фазой, зависящей от turnover
    if avg_range != 0 and avg_volume != 0:
        phase = sigmoid(avg_price_change / avg_range + (turnover / avg_volume - 1)) * 2 * math.pi
    else:
        phase = 0
    interference_factor = 0.15 * cmath.exp(1j * phase)
    if avg_price_change > 0:
        growth_amp += interference_factor
    elif avg_price_change < 0:
        decline_amp += interference_factor

    # Запутанность с учётом volume
    entanglement_boost = abs(trend_strength) * (volume / avg_volume) * 0.2
    if trend_strength > 0:
        growth_amp += entanglement_boost * cmath.exp(1j * phase)
        decline_amp -= entanglement_boost / 2
    elif trend_strength < 0:
        decline_amp += entanglement_boost * cmath.exp(1j * phase)
        growth_amp -= entanglement_boost / 2

    # Учёт волатильности
    if range_percent < 0.05 or range_price < (avg_range * 0.8):
        stagnation_amp += 0.3
    elif range_percent > 0.2 or range_price > (avg_range * 1.2):
        growth_amp += 0.15
        decline_amp += 0.15

    # Нормализация
    norm = math.sqrt(abs(growth_amp)**2 + abs(decline_amp)**2 + abs(stagnation_amp)**2)
    if norm > 0:
        growth_amp /= norm
        decline_amp /= norm
        stagnation_amp /= norm

    # Вероятности
    growth_prob = abs(growth_amp)**2
    decline_prob = abs(decline_amp)**2
    stagnation_prob = abs(stagnation_amp)**2

    return (growth_amp, decline_amp, stagnation_amp), (growth_prob, decline_prob, stagnation_prob), avg_range

def predict_close_price(df, current_idx, prev_amps=None, lookback=5):
    """Предсказывает closePrice с эволюцией и коллапсом."""
    row = df.iloc[current_idx]
    
    # Расчёт волновой функции
    amps, probs, avg_range = calculate_wave_function(df, current_idx, prev_amps, lookback)
    growth_amp, decline_amp, stagnation_amp = amps
    growth_prob, decline_prob, stagnation_prob = probs

    # Оценка движения
    current_range = row["highPrice"] - row["lowPrice"]
    price_move = (current_range + avg_range) / 2 * 0.75
    growth_price = row["closePrice"] + price_move
    decline_price = row["closePrice"] - price_move
    stagnation_price = row["closePrice"]

    # Ожидаемое значение
    expected_close = (growth_prob * growth_price) + \
                     (decline_prob * decline_price) + \
                     (stagnation_prob * stagnation_price)

    # Коллапс
    rand = random.random()
    if rand < growth_prob:
        collapsed_price = growth_price
    elif rand < (growth_prob + decline_prob):
        collapsed_price = decline_price
    else:
        collapsed_price = stagnation_price

    return expected_close, collapsed_price, amps

# Загрузка данных из CSV
csv_file = "data/kline/BTCUSDT/1/2025-02-14/01.csv"  # Укажи имя своего файла
df = pd.read_csv(csv_file)

# Проверка колонок
required_cols = ["startTime", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "turnover"]
if not all(col in df.columns for col in required_cols):
    raise ValueError("CSV-файл должен содержать колонки: " + ", ".join(required_cols))

# Прогноз и сбор данных
expected_prices = []
collapsed_prices = []
actual_prices = df["closePrice"].tolist()[1:]
times = df["startTime"].tolist()[1:]
prev_amps = None

for i in range(len(df) - 1):
    expected_close, collapsed_close, prev_amps = predict_close_price(df, i, prev_amps, lookback=5)
    expected_prices.append(expected_close)
    collapsed_prices.append(collapsed_close)

# Расчёт ошибок
if expected_prices and actual_prices:
    mse_exp = sum((pred - actual) ** 2 for pred, actual in zip(expected_prices, actual_prices)) / len(expected_prices)
    rmse_exp = math.sqrt(mse_exp)
    mse_col = sum((pred - actual) ** 2 for pred, actual in zip(collapsed_prices, actual_prices)) / len(collapsed_prices)
    rmse_col = math.sqrt(mse_col)
    print(f"Ошибка модели (RMSE, ожидаемое): {rmse_exp:.2f} USD")
    print(f"Ошибка модели (RMSE, после коллапса): {rmse_col:.2f} USD")
else:
    print("Недостаточно данных для расчёта ошибки")

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(times, actual_prices, label="Реальный closePrice", color="blue", marker="o")
plt.plot(times, expected_prices, label="Ожидаемый closePrice", color="orange", linestyle="--", marker="x")
plt.plot(times, collapsed_prices, label="Коллапсированный closePrice", color="green", linestyle="-.", marker="^")
plt.xlabel("Время (startTime)")
plt.ylabel("Цена (USD)")
plt.title("Сравнение реальных, ожидаемых и коллапсированных цен BTC")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
