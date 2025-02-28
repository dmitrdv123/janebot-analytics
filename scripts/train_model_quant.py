import pandas as pd
import matplotlib.pyplot as plt
import math
import cmath
import random
import numpy as np
from scipy.linalg import expm, eigvals
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import load_data

def sigmoid(x):
    """Сигмоидная функция для нормализации."""
    return 1 / (1 + math.exp(-x))

def compute_hamiltonian(df, current_idx, timeframe=1, lookback=100):
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
    
    T = volatility_factor / volume_factor if volume_factor != 0 else 1
    T_c = 1
    
    # Бозонные энергии (осцилляторы)
    omega_growth = volatility_factor * volume_factor
    omega_decline = volatility_factor * (1 - volume_factor)
    omega_stagnation = turnover_factor

    # H 3x3
    H = np.zeros((3, 3), dtype=complex)

    H[0, 0] = omega_growth / 2
    H[1, 1] = omega_decline / 2
    H[2, 2] = omega_stagnation / 2
    
    tunneling_factor = 0.1 * volume_factor * volatility_factor
    H[0, 1] = H[1, 0] = tunneling_factor
    H[0, 2] = H[2, 0] = tunneling_factor / 2
    H[1, 2] = H[2, 1] = tunneling_factor / 2

    eta = 0.005
    vacuum_fluct = eta * np.random.uniform(-1, 1, (3, 3)) * math.exp(-turnover_factor)
    H += vacuum_fluct + vacuum_fluct.conj().T

    noise_amplitude = 0.02 * (1 / math.log(timeframe + 1)) * (1.2 if T > T_c else 1)
    noise = np.random.normal(0, noise_amplitude, (3, 3)) + 1j * np.random.normal(0, noise_amplitude, (3, 3))
    H += noise
    H = (H + H.conj().T) / 2

    return H / 1000

def compute_correlator(psi_current, psi_past):
    psi_c = np.array(psi_current, dtype=complex)
    psi_p = np.array(psi_past, dtype=complex)
    correlator = abs(np.vdot(psi_c, psi_p)) ** 2
    return min(correlator, 1.0)  # Ограничиваем до 1 из-за численных ошибок

def compute_entropy(ensemble_amps):
    rho = np.zeros((3, 3), dtype=complex)
    for amps in ensemble_amps:
        psi = np.array(amps, dtype=complex)
        rho += np.outer(psi, psi.conj())
    rho /= len(ensemble_amps)
    eigenvalues = eigvals(rho)
    eigenvalues = eigenvalues.real  # Берем только действительную часть
    eigenvalues = np.clip(eigenvalues, 1e-10, 1)  # Избегаем log(0)
    S = -sum(eig * math.log(eig) for eig in eigenvalues if eig > 0)
    return S

def holographic_convolution(df, current_idx, prev_states, lookback=100):
    if not prev_states or current_idx <= 0:
        return None
    holographic_amps = np.zeros(3, dtype=complex)  # 3 для SUSY
    total_weight = 0
    beta = 0.05
    d = 1
    
    start_idx = max(0, current_idx - lookback)
    for k in range(start_idx, current_idx):
        if k not in prev_states:
            continue
        trend_strength = np.corrcoef(df["startTime"][max(0, k-lookback):k+1], 
                                     df["closePrice"][max(0, k-lookback):k+1])[0, 1] if k > 0 else 0
        distance = current_idx - k
        weight = abs(trend_strength) / (distance ** d) * math.exp(-beta * distance)
        holographic_amps += np.array(prev_states[k], dtype=complex) * weight
        total_weight += weight
    
    if total_weight > 0:
        holographic_amps /= total_weight
        norm = math.sqrt(sum(abs(x)**2 for x in holographic_amps))
        if norm > 0:
            holographic_amps /= norm
        return tuple(holographic_amps)
    return None

def tomography_initial_state(df, current_idx, lookback=100):
    if current_idx <= 0:
        return tuple([complex(1 / math.sqrt(3), 0)] * 3)  # 3 для SUSY
    
    start_idx = max(0, current_idx - lookback)
    hist_df = df.iloc[start_idx:current_idx]
    if not hist_df.empty:
        delta_p = df["closePrice"].iloc[current_idx-1] - df["closePrice"].iloc[current_idx-2] if current_idx > 1 else 0
        growth_prob = max(delta_p / 100, 0)  # Нормализация на 100 USD
        decline_prob = max(-delta_p / 100, 0)
        stagnation_prob = math.exp(-abs(delta_p) / 100)
        
        total = growth_prob + decline_prob + stagnation_prob
        if total > 0:
            growth_prob /= total
            decline_prob /= total
            stagnation_prob /= total
        return (complex(math.sqrt(growth_prob), 0), 
                complex(math.sqrt(decline_prob), 0), 
                complex(math.sqrt(stagnation_prob), 0))
    return tuple([complex(1 / math.sqrt(3), 0)] * 3)

def heat_engine_boost(turnover_factor):
    T_hot = max(turnover_factor, 1 / turnover_factor if turnover_factor > 0 else 1)
    T_cold = min(turnover_factor, 1 / turnover_factor if turnover_factor > 0 else 1)
    eta = 1 - T_cold / T_hot if T_hot > 0 else 0
    return 0.1 * eta * turnover_factor

def evolve_wave_function(prev_amps, H, turnover_factor, volume_factor, T, T_c, avg_price_change, avg_range, dt=1):
    psi = np.array(prev_amps, dtype=complex)
    U = expm(-1j * H * dt)
    
    D = 0.01 * turnover_factor
    diffusion = D * (psi - np.roll(psi, 1))
    psi = U @ psi - 1j * diffusion
    
    # Тепловая машина
    heat_boost = heat_engine_boost(turnover_factor)
    psi += heat_boost * psi  # Усиление амплитуд через энергию
    
    norm = math.sqrt(sum(abs(x)**2 for x in psi))
    if norm > 0:
        psi /= norm
    
    return tuple(psi)

def calculate_wave_function(df, current_idx, prev_states, prev_error=0, timeframe=1, lookback=100):
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
        avg_turnover = hist_df["turnover"].mean()
        trend_strength = np.corrcoef(hist_df["startTime"], hist_df["closePrice"])[0, 1] if len(hist_df) > 1 else 0
    else:
        avg_price_change = price_change
        avg_range = range_price
        avg_volume = volume
        avg_turnover = turnover
        trend_strength = 0

    turnover_factor = turnover / avg_turnover if avg_turnover != 0 else 1
    volume_factor = volume / avg_volume if avg_volume != 0 else 1
    T = (range_price / avg_range if avg_range != 0 else 1) / volume_factor
    T_c = 1

    # Используем prev_states как словарь, проверяем наличие предыдущего состояния
    if current_idx == 0 or current_idx - 1 not in prev_states:
        growth_amp, decline_amp, stagnation_amp = tomography_initial_state(df, current_idx, lookback)
    else:
        H = compute_hamiltonian(df, current_idx - 1, timeframe, lookback)
        growth_amp, decline_amp, stagnation_amp = evolve_wave_function(
            prev_states[current_idx - 1], H, turnover_factor, volume_factor, T, T_c, avg_price_change, avg_range
        )
        holographic_amps = holographic_convolution(df, current_idx, prev_states, lookback)
        if holographic_amps:
            growth_amp = 0.5 * growth_amp + 0.5 * holographic_amps[0]
            decline_amp = 0.5 * decline_amp + 0.5 * holographic_amps[1]
            stagnation_amp = 0.5 * stagnation_amp + 0.5 * holographic_amps[2]

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

    if avg_range != 0 and avg_volume != 0:
        phase = sigmoid(avg_price_change / avg_range + (turnover / avg_volume - 1) + 0.01 * prev_error) * 2 * math.pi
    else:
        phase = 0
    interference_factor = 0.05 * cmath.exp(1j * phase)
    if avg_price_change > 0:
        growth_amp += interference_factor
    elif avg_price_change < 0:
        decline_amp += interference_factor

    entanglement_boost = abs(trend_strength) * (volume / avg_volume) * 0.05
    if trend_strength > 0:
        growth_amp += entanglement_boost * cmath.exp(1j * phase)
        decline_amp -= entanglement_boost / 2
    elif trend_strength < 0:
        decline_amp += entanglement_boost * cmath.exp(1j * phase)
        growth_amp -= entanglement_boost / 2

    if range_percent < 0.05 or range_price < (avg_range * 0.8):
        stagnation_amp += 0.3
    elif range_percent > 0.2 or range_price > (avg_range * 1.2):
        growth_amp += 0.15
        decline_amp += 0.15

    # Корреляторы с прошлыми состояниями
    correlator_sum = 0
    correlator_count = 0
    for k in range(start_idx, current_idx):
        if k in prev_states:
            correlator = compute_correlator(
                (growth_amp, decline_amp, stagnation_amp),
                prev_states[k]
            )
            correlator_sum += correlator
            correlator_count += 1
    avg_correlator = correlator_sum / correlator_count if correlator_count > 0 else 0.5

    # Усиление амплитуд пропорционально корреляции
    correlation_boost = 0.1 * avg_correlator
    growth_amp *= (1 + correlation_boost)
    decline_amp *= (1 + correlation_boost)
    stagnation_amp *= (1 + correlation_boost)

    amps = np.array([growth_amp, decline_amp, stagnation_amp], dtype=complex)
    norm = math.sqrt(sum(abs(x)**2 for x in amps))
    if norm > 0:
        amps /= norm
    growth_amp, decline_amp, stagnation_amp = amps

    # Энтропия фон Неймана
    entropy = compute_entropy(amps)
    entropy_factor = 1 - min(entropy / math.log(3), 1)

    growth_prob = abs(growth_amp)**2
    decline_prob = abs(decline_amp)**2
    stagnation_prob = abs(stagnation_amp)**2
    total = growth_prob + decline_prob + stagnation_prob
    if total > 0:
        growth_prob /= total
        decline_prob /= total
        stagnation_prob /= total
    # Корректировка вероятностей с коррелятором
    growth_prob *= (1 + 0.1 * avg_correlator)
    decline_prob *= (1 + 0.1 * avg_correlator)
    stagnation_prob *= (1 + 0.1 * avg_correlator)
    total = growth_prob + decline_prob + stagnation_prob
    if total > 0:
        growth_prob /= total
        decline_prob /= total
        stagnation_prob /= total

    return (growth_amp, decline_amp, stagnation_amp), (growth_prob, decline_prob, stagnation_prob), avg_range, entropy_factor, avg_correlator

def predict_close_price(df, current_idx, prev_states, prev_error=0, timeframe=1, ensemble_size=100):
    """Предсказывает closePrice."""
    lookback = 100
    row = df.iloc[current_idx]
    
    # Расчёт волновой функции
    amps, probs, avg_range, entropy_factor, mean_correlator = calculate_wave_function(
        df, current_idx, prev_states, prev_error, timeframe, lookback
    )
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
    # Сужаем интервалы пропорционально надёжности (1 - энтропия)
    ci_lower = collapsed_mean - 1.96 * collapsed_std * entropy_factor
    ci_upper = collapsed_mean + 1.96 * collapsed_std * entropy_factor

    # Логика бота
    signal = None
    rmse_threshold = 50
    current_close = row["closePrice"]
    expected_change = abs(expected_close - current_close)
    
    # Фильтр по волатильности (средний размах за последние 10 свечей)
    vol_lookback = min(10, current_idx)
    recent_volatility = (df["highPrice"].iloc[max(0, current_idx-vol_lookback):current_idx+1] - 
                         df["lowPrice"].iloc[max(0, current_idx-vol_lookback):current_idx+1]).mean()

    if expected_change > rmse_threshold and mean_correlator > 0.7 and recent_volatility > 50:  # Фильтр по коррелятору
        if ci_lower > current_close:
            signal = "Long"
        elif ci_upper < current_close:
            signal = "Short"

    return expected_close, collapsed_mean, ci_lower, ci_upper, amps, mean_correlator, signal

# Загрузка данных
symbol = 'BTCUSDT'
interval = '1'  # Kline interval (1m, 5m, 15m, etc.)
df_kline = load_data(f'data/kline/{symbol}/{interval}')

# Проверка колонок
required_cols = ["startTime", "openPrice", "highPrice", "lowPrice", "closePrice", "volume", "turnover"]
if not all(col in df_kline.columns for col in required_cols):
    raise ValueError("CSV-файл Kline должен содержать колонки: " + ", ".join(required_cols))

# Прогноз квантовой модели и тестирование бота
fee_open = 0.0002
fee_close = 0.0002
timeframe = 1
ensemble_size = 100  # Увеличено с 20
random.seed(42)  # Фиксация случайности для стабильности
expected_prices = []
collapsed_prices = []
ci_lowers = []
ci_uppers = []
actual_prices = df_kline["closePrice"].tolist()[1:-1]  # Убрали последнюю свечу
times = df_kline["startTime"].tolist()[:-2]  # Убрали последние две свечи
correlators = []
signals = []
prev_states = {}
prev_error = 0

long_signals = 0
short_signals = 0
long_correct = 0
short_correct = 0
total_profit = 0

for i in range(len(df_kline) - 2):  # Ограничен до len(df) - 2
    expected_close, collapsed_mean, ci_lower, ci_upper, amps, mean_correlator, signal = predict_close_price(
        df_kline, i, prev_states, prev_error, timeframe, ensemble_size=ensemble_size
    )
    expected_prices.append(expected_close)
    collapsed_prices.append(collapsed_mean)
    ci_lowers.append(ci_lower)
    ci_uppers.append(ci_upper)
    correlators.append(mean_correlator)
    signals.append(signal)
    prev_states[i] = amps
    prev_error = actual_prices[i] - collapsed_mean if i < len(actual_prices) else 0

    # Анализ сигналов бота с учётом стоп-лосса и тейк-профита
    if signal == "Long":
        long_signals += 1
        next_price = df_kline["closePrice"].iloc[i + 1]
        entry_price = df_kline["closePrice"].iloc[i]
        profit = next_price - entry_price - fee_open * entry_price - fee_close * next_price
        stop_loss = max(-0.5 * abs(expected_close - entry_price), -30)
        if profit < -30:
            profit = -30 - fee_open * entry_price - fee_close * next_price  # Стоп-лосс
        total_profit += profit
        if next_price > entry_price:
            long_correct += 1
    elif signal == "Short":
        short_signals += 1
        next_price = df_kline["closePrice"].iloc[i + 1]
        entry_price = df_kline["closePrice"].iloc[i]
        profit = entry_price - next_price - fee_open * entry_price - fee_close * next_price
        stop_loss = max(-0.5 * abs(expected_close - entry_price), -30)
        if profit < -30:
            profit = -30 - fee_open * entry_price - fee_close * next_price # Стоп-лосс
        total_profit += profit
        if next_price < entry_price:
            short_correct += 1

# Расчёт ошибок и метрик
if expected_prices and actual_prices:
    mse_exp = mean_squared_error(actual_prices, expected_prices)
    rmse_exp = math.sqrt(mse_exp)
    mae_exp = mean_absolute_error(actual_prices, expected_prices)
    mse_col = mean_squared_error(actual_prices, collapsed_prices)
    rmse_col = math.sqrt(mse_col)
    mae_col = mean_absolute_error(actual_prices, collapsed_prices)
    print(f"Ошибка квантовой модели (RMSE, ожидаемое): {rmse_exp:.2f} USD")
    print(f"Ошибка квантовой модели (RMSE, ансамбль): {rmse_col:.2f} USD")
    print(f"Mean Absolute Error (MAE, ожидаемое): {mae_exp:.2f} USD")
    print(f"Mean Absolute Error (MAE, ансамбль): {mae_col:.2f} USD")

    avg_candle_range = (df_kline["highPrice"] - df_kline["lowPrice"]).mean()
    print(f"Средний размах свечей (highPrice - lowPrice): {avg_candle_range:.2f} USD")

    price_changes = [abs(df_kline["closePrice"].iloc[i] - df_kline["closePrice"].iloc[i-1]) for i in range(1, len(df_kline))]
    avg_price_change = np.mean(price_changes)
    print(f"Среднее абсолютное изменение цены (|closePrice[i] - closePrice[i-1]|): {avg_price_change:.2f} USD")

    std_close_price = df_kline["closePrice"].std()
    print(f"Стандартное отклонение closePrice: {std_close_price:.2f} USD")

    # Результаты бота
    total_signals = long_signals + short_signals
    print(f"\nРезультаты бота:")
    print(f"Всего сигналов: {total_signals}")
    print(f"Сигналов Long: {long_signals}")
    print(f"Точных Long: {long_correct} ({long_correct/long_signals*100:.2f}%)" if long_signals > 0 else f"Точных Long: {long_correct} (0.00%)")
    print(f"Сигналов Short: {short_signals}")
    print(f"Точных Short: {short_correct} ({short_correct/short_signals*100:.2f}%)" if short_signals > 0 else f"Точных Short: {short_correct} (0.00%)")
    total_correct = long_correct + short_correct
    accuracy = total_correct / total_signals * 100 if total_signals > 0 else 0
    print(f"Общая точность: {accuracy:.2f}%")
    print(f"Общая прибыль/убыток: {total_profit:.2f} USD")
    avg_profit_per_trade = total_profit / total_signals if total_signals > 0 else 0
    print(f"Средняя прибыль на сделку: {avg_profit_per_trade:.2f} USD")
else:
    print("Недостаточно данных для расчёта ошибки")

# Построение графиков
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# График цен
ax1.plot(times, actual_prices, label="Реальный closePrice", color="blue", marker="o")
ax1.plot(times, expected_prices, label="Ожидаемый closePrice (квант)", color="orange", linestyle="--", marker="x")
ax1.plot(times, collapsed_prices, label="Ансамбль closePrice (квант)", color="green", linestyle="-.", marker="^")
ax1.fill_between(times, ci_lowers, ci_uppers, color="green", alpha=0.2, label="95% Доверительный интервал (с энтропией)")
ax1.set_ylabel("Цена (USD)")
ax1.set_title(f"Сравнение цен (timeframe={timeframe} мин, ensemble_size={ensemble_size}, lookback=100))")
ax1.legend()
ax1.grid(True)

# График корреляторов
ax2.plot(times, correlators, label="Средний коррелятор", color="purple", linestyle="-", marker="o")
ax2.set_xlabel("Время (startTime)")
ax2.set_ylabel("Средний коррелятор")
ax2.set_title("Корреляция с прошлым")
ax2.legend()
ax2.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
