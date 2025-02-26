import pandas as pd
import matplotlib.pyplot as plt
import math
import cmath
import random
import numpy as np
from scipy.linalg import expm

from utils import load_data

def sigmoid(x):
    """Сигмоидная функция для нормализации."""
    return 1 / (1 + math.exp(-x))

def compute_hamiltonian(df, current_idx, timeframe=1, lookback=5):
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

    volatility_factor = range_price / avg_range if avg_range != 0 else 1
    volume_factor = volume / avg_volume if avg_volume != 0 else 1
    turnover_factor = turnover / avg_turnover if avg_turnover != 0 else 1
    
    h_growth = volatility_factor * volume_factor
    h_decline = volatility_factor * (2 - volume_factor)
    h_stagnation = turnover_factor

    H = np.zeros((3, 3), dtype=complex)
    H[0, 0] = h_growth
    H[1, 1] = h_decline
    H[2, 2] = h_stagnation

    interaction_strength = 0.1 * volatility_factor
    H[0, 1] = H[1, 0] = interaction_strength
    H[0, 2] = H[2, 0] = interaction_strength / 2
    H[1, 2] = H[2, 1] = interaction_strength / 2

    noise_amplitude = 0.05 * (1 / math.log(timeframe + 1))
    noise = np.random.normal(0, noise_amplitude, (3, 3)) + 1j * np.random.normal(0, noise_amplitude, (3, 3))
    H += noise
    H = (H + H.conj().T) / 2

    return H / 1000

def evolve_wave_function(prev_amps, H, dt=1):
    psi = np.array(prev_amps, dtype=complex)
    U = expm(-1j * H * dt)
    new_psi = U @ psi
    
    norm = math.sqrt(sum(abs(x)**2 for x in new_psi))
    if norm > 0:
        new_psi /= norm
    
    return tuple(new_psi)

def calculate_wave_function(df, current_idx, prev_amps=None, timeframe=1, lookback=5):
    """Рассчитывает волновую функцию."""
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
        H = compute_hamiltonian(df, current_idx - 1, timeframe, lookback)
        growth_amp, decline_amp, stagnation_amp = evolve_wave_function(prev_amps, H)

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
    amps = np.array([growth_amp, decline_amp, stagnation_amp], dtype=complex)
    norm = math.sqrt(sum(abs(x)**2 for x in amps))
    if norm > 0:
        amps /= norm
    growth_amp, decline_amp, stagnation_amp = amps

    # Вероятности
    growth_prob = abs(growth_amp)**2
    decline_prob = abs(decline_amp)**2
    stagnation_prob = abs(stagnation_amp)**2

    return (growth_amp, decline_amp, stagnation_amp), (growth_prob, decline_prob, stagnation_prob), avg_range

def predict_close_price(df, current_idx, prev_amps=None, timeframe=1, ensemble_size=10):
    """Предсказывает closePrice."""
    lookback = max(5, int(60 / timeframe))
    row = df.iloc[current_idx]
    
    # Расчёт волновой функции
    amps, probs, avg_range = calculate_wave_function(df, current_idx, prev_amps, timeframe, lookback)
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
    collapsed_ensemble = []
    for _ in range(ensemble_size):
        rand = random.random()
        if rand < growth_prob:
            collapsed_ensemble.append(growth_price)
        elif rand < (growth_prob + decline_prob):
            collapsed_ensemble.append(decline_price)
        else:
            collapsed_ensemble.append(stagnation_price)
    
    collapsed_mean = np.mean(collapsed_ensemble)
    collapsed_std = np.std(collapsed_ensemble)
    ci_lower = collapsed_mean - 1.96 * collapsed_std
    ci_upper = collapsed_mean + 1.96 * collapsed_std

    return expected_close, collapsed_mean, ci_lower, ci_upper, amps

# Загрузка данных
symbol = 'BTCUSDT'
interval = '1'  # Kline interval (1m, 5m, 15m, etc.)
df = load_data(f'data/kline/{symbol}/{interval}')

# Проверка колонок
required_cols = ["startTime", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "turnover"]
if not all(col in df.columns for col in required_cols):
    raise ValueError("CSV-файл должен содержать колонки: " + ", ".join(required_cols))

# Прогноз квантовой модели
timeframe = 1
ensemble_size = 20
expected_prices = []
collapsed_prices = []
ci_lowers = []
ci_uppers = []
actual_prices = df["closePrice"].tolist()[1:]
times = df["startTime"].tolist()[:-1]  # Время для предсказания — момент начала свечи
prev_amps = None

for i in range(len(df) - 1):
    expected_close, collapsed_mean, ci_lower, ci_upper, prev_amps = predict_close_price(
        df, i, prev_amps, timeframe, ensemble_size=ensemble_size
    )
    expected_prices.append(expected_close)
    collapsed_prices.append(collapsed_mean)
    ci_lowers.append(ci_lower)
    ci_uppers.append(ci_upper)

# Расчёт ошибок
if expected_prices and actual_prices:
    mse_exp = sum((pred - actual) ** 2 for pred, actual in zip(expected_prices, actual_prices)) / len(expected_prices)
    rmse_exp = math.sqrt(mse_exp)
    mse_col = sum((pred - actual) ** 2 for pred, actual in zip(collapsed_prices, actual_prices)) / len(collapsed_prices)
    rmse_col = math.sqrt(mse_col)
    print(f"Ошибка квантовой модели (RMSE, ожидаемое): {rmse_exp:.2f} USD")
    print(f"Ошибка квантовой модели (RMSE, ансамбль): {rmse_col:.2f} USD")
else:
    print("Недостаточно данных для расчёта ошибки")

# Построение графика
plt.figure(figsize=(14, 7))
plt.plot(times, actual_prices, label="Реальный closePrice", color="blue", marker="o")
plt.plot(times, expected_prices, label="Ожидаемый closePrice (квант)", color="orange", linestyle="--", marker="x")
plt.plot(times, collapsed_prices, label="Ансамбль closePrice (квант)", color="green", linestyle="-.", marker="^")
plt.fill_between(times, ci_lowers, ci_uppers, color="green", alpha=0.2, label="95% Доверительный интервал")
plt.xlabel("Время (startTime)")
plt.ylabel("Цена (USD)")
plt.title(f"Сравнение моделей (timeframe={timeframe} мин, ensemble_size={ensemble_size})")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
