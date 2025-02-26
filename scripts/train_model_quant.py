import matplotlib.pyplot as plt
import math
import cmath
import random
import numpy as np
from scipy.linalg import expm, eigvals

from utils import load_data

def sigmoid(x):
    """Сигмоидная функция для нормализации."""
    return 1 / (1 + math.exp(-x))

def compute_hamiltonian(df, current_idx, timeframe=1, lookback=5, noise_seed=None):
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
        dp_dt = (hist_df["closePrice"].iloc[-1] - hist_df["closePrice"].iloc[-2]) / timeframe if len(hist_df) > 1 else 0
    else:
        avg_range = range_price
        avg_volume = volume
        avg_turnover = turnover
        dp_dt = 0

    volatility_factor = range_price / avg_range if avg_range != 0 else 1
    volume_factor = volume / avg_volume if avg_volume != 0 else 1
    turnover_factor = turnover / avg_turnover if avg_turnover != 0 else 1
    
    T = volatility_factor / volume_factor if volume_factor != 0 else 1
    T_c = 1
    
    # Бозонные энергии (осцилляторы)
    omega_growth = volatility_factor * volume_factor
    omega_decline = volatility_factor * (1 - volume_factor)
    omega_stagnation = turnover_factor
    
    # Фермионные энергии (спины, зависят от dp/dt)
    fermi_growth = abs(dp_dt) if dp_dt > 0 else 0
    fermi_decline = abs(dp_dt) if dp_dt < 0 else 0
    fermi_stagnation = 1 / (1 + abs(dp_dt)) if abs(dp_dt) > 0 else 1

    # H 6x6
    H = np.zeros((6, 6), dtype=complex)
    # Бозоны
    H[0, 0] = omega_growth / 2  # Рост
    H[1, 1] = omega_decline / 2  # Падение
    H[2, 2] = omega_stagnation / 2  # Стагнация
    # Фермионы
    H[3, 3] = fermi_growth / 2
    H[4, 4] = fermi_decline / 2
    H[5, 5] = fermi_stagnation / 2
    
    # Взаимодействия бозон-бозон
    J = 0.2 * volume_factor * (1 if T < T_c else 0.5)
    H[0, 1] = H[1, 0] = -J
    H[0, 2] = H[2, 0] = -J / 2
    H[1, 2] = H[2, 1] = -J / 2
    
    # Взаимодействия бозон-фермион
    int_strength = 0.1 * volatility_factor
    H[0, 3] = H[3, 0] = int_strength  # Рост <-> Тенденция роста
    H[1, 4] = H[4, 1] = int_strength  # Падение <-> Тенденция падения
    H[2, 5] = H[5, 2] = int_strength  # Стагнация <-> Тенденция стагнации

    # Флуктуации вакуума
    eta = 0.01  # Амплитуда флуктуаций
    np.random.seed(noise_seed)  # Разный шум для каждой траектории ансамбля
    vacuum_fluct = eta * np.random.uniform(-1, 1, (6, 6)) * math.exp(-turnover_factor)
    H += vacuum_fluct + vacuum_fluct.conj().T  # Сохраняем hermiticity

    noise_amplitude = 0.05 * (1 / math.log(timeframe + 1)) * (1.2 if T > T_c else 1)
    noise = np.random.normal(0, noise_amplitude, (6, 6)) + 1j * np.random.normal(0, noise_amplitude, (6, 6))
    H += noise
    H = (H + H.conj().T) / 2

    return H / 1000

def compute_correlator(psi_current, psi_past):
    psi_c = np.array(psi_current, dtype=complex)
    psi_p = np.array(psi_past, dtype=complex)
    correlator = abs(np.vdot(psi_c, psi_p)) ** 2
    return min(correlator, 1.0)  # Ограничиваем до 1 из-за численных ошибок

def compute_entropy(ensemble_amps):
    rho = np.zeros((6, 6), dtype=complex)
    for amps in ensemble_amps:
        psi = np.array(amps, dtype=complex)
        rho += np.outer(psi, psi.conj())
    rho /= len(ensemble_amps)
    eigenvalues = eigvals(rho)
    eigenvalues = eigenvalues.real  # Берем только действительную часть
    eigenvalues = np.clip(eigenvalues, 1e-10, 1)  # Избегаем log(0)
    S = -sum(eig * math.log(eig) for eig in eigenvalues if eig > 0)
    return S

def holographic_convolution(df, current_idx, prev_states, lookback=5):
    if not prev_states or current_idx <= 0:
        return None
    holographic_amps = np.zeros(6, dtype=complex)  # 6 для SUSY
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

def tomography_initial_state(df, current_idx, lookback=5):
    if current_idx <= 0:
        return tuple([complex(1 / math.sqrt(6), 0)] * 6)  # 6 для SUSY
    
    start_idx = max(0, current_idx - lookback)
    hist_df = df.iloc[start_idx:current_idx]
    if not hist_df.empty:
        delta_p = df["closePrice"].iloc[current_idx-1] - df["closePrice"].iloc[current_idx-2] if current_idx > 1 else 0
        dp_dt = delta_p / timeframe if timeframe > 0 else 0
        growth_prob = max(delta_p / 100, 0)  # Нормализация на 100 USD
        decline_prob = max(-delta_p / 100, 0)
        stagnation_prob = math.exp(-abs(delta_p) / 100)
        fermi_growth_prob = max(dp_dt / 100, 0)
        fermi_decline_prob = max(-dp_dt / 100, 0)
        fermi_stagnation_prob = math.exp(-abs(dp_dt) / 100)
        
        total_b = growth_prob + decline_prob + stagnation_prob
        total_f = fermi_growth_prob + fermi_decline_prob + fermi_stagnation_prob
        if total_b > 0 and total_f > 0:
            growth_prob /= total_b
            decline_prob /= total_b
            stagnation_prob /= total_b
            fermi_growth_prob /= total_f
            fermi_decline_prob /= total_f
            fermi_stagnation_prob /= total_f
        return (complex(math.sqrt(growth_prob), 0), 
                complex(math.sqrt(decline_prob), 0), 
                complex(math.sqrt(stagnation_prob), 0),
                complex(math.sqrt(fermi_growth_prob), 0),
                complex(math.sqrt(fermi_decline_prob), 0),
                complex(math.sqrt(fermi_stagnation_prob), 0))
    return tuple([complex(1 / math.sqrt(6), 0)] * 6)

def vortex_operator(turnover_factor, avg_price_change, avg_range):
    theta_vortex = 0.1 * turnover_factor * math.sin(avg_price_change / avg_range if avg_range != 0 else 0)
    V = np.diag([cmath.exp(1j * theta_vortex), cmath.exp(-1j * theta_vortex), 1, 1, 1, 1], k=0)
    return V

def heat_engine_boost(turnover_factor):
    T_hot = max(turnover_factor, 1 / turnover_factor if turnover_factor > 0 else 1)
    T_cold = min(turnover_factor, 1 / turnover_factor if turnover_factor > 0 else 1)
    eta = 1 - T_cold / T_hot if T_hot > 0 else 0
    return 0.1 * eta * turnover_factor  # Преобразование энергии в амплитуды

def evolve_wave_function(prev_amps, H, turnover_factor, volume_factor, T, T_c, avg_price_change, avg_range, dt=1):
    psi = np.array(prev_amps, dtype=complex)
    U = expm(-1j * H * dt)
    
    gamma = (1 / volume_factor if volume_factor != 0 else 1) * math.exp(T - T_c)
    decoherence_factor = math.exp(-gamma * dt)
    psi *= decoherence_factor
    
    # Вихри
    V = vortex_operator(turnover_factor, avg_price_change, avg_range)
    psi = V @ psi
    
    psi = U @ psi
    
    # Тепловая машина
    heat_boost = heat_engine_boost(turnover_factor)
    psi += heat_boost * psi  # Усиление амплитуд через энергию
    
    norm = math.sqrt(sum(abs(x)**2 for x in psi))
    if norm > 0:
        psi /= norm
    
    return tuple(psi)

def calculate_cell_state(df, cell_start_idx, cell_end_idx, prev_cell_amps=None, timeframe=1, lookback=5):
    # Локальная эволюция для ячейки
    cell_amps = prev_cell_amps if prev_cell_amps is not None else tomography_initial_state(df, cell_start_idx, lookback)
    for idx in range(cell_start_idx, min(cell_end_idx, len(df))):
        H = compute_hamiltonian(df, idx, timeframe, lookback)
        row = df.iloc[idx]
        turnover_factor = row["turnover"] / df["turnover"][max(0, idx-lookback):idx+1].mean() if df["turnover"][max(0, idx-lookback):idx+1].mean() != 0 else 1
        volume_factor = row["volume"] / df["volume"][max(0, idx-lookback):idx+1].mean() if df["volume"][max(0, idx-lookback):idx+1].mean() != 0 else 1
        T = (row["highPrice"] - row["lowPrice"]) / df["highPrice"][max(0, idx-lookback):idx+1].mean() if df["highPrice"][max(0, idx-lookback):idx+1].mean() != 0 else 1 / volume_factor
        T_c = 1
        avg_price_change = df["closePrice"][max(0, idx-lookback):idx+1].mean()
        avg_range = (df["highPrice"][max(0, idx-lookback):idx+1] - df["lowPrice"][max(0, idx-lookback):idx+1]).mean()
        cell_amps = evolve_wave_function(cell_amps, H, turnover_factor, volume_factor, T, T_c, avg_price_change, avg_range)
    return cell_amps

def calculate_wave_function(df, current_idx, prev_states=None, prev_error=0, timeframe=1, lookback=5, entangle_range=3, ensemble_size_traj=5, cell_size=5):
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

    # Квантовые ячейки
    cell_idx = current_idx // cell_size
    cell_start = cell_idx * cell_size
    cell_end = min(cell_start + cell_size, len(df))
    prev_cell_idx = max(0, cell_idx - 1)
    prev_cell_start = prev_cell_idx * cell_size
    prev_cell_amps = prev_states.get(prev_cell_start, None) if prev_states else None
    cell_amps = calculate_cell_state(df, cell_start, cell_end, prev_cell_amps, timeframe, lookback)

    ensemble_amps = []
    ensemble_correlators = []
    for traj in range(ensemble_size_traj):
        if prev_states is None or current_idx not in prev_states:
            growth_amp, decline_amp, stagnation_amp, fg_amp, fd_amp, fs_amp = tomography_initial_state(df, current_idx, lookback)
        else:
            H = compute_hamiltonian(df, current_idx - 1, timeframe, lookback, noise_seed=traj)
            growth_amp, decline_amp, stagnation_amp, fg_amp, fd_amp, fs_amp = evolve_wave_function(
                prev_states[current_idx - 1], H, turnover_factor, volume_factor, T, T_c, avg_price_change, avg_range
            )
            holographic_amps = holographic_convolution(df, current_idx, prev_states, lookback)
            if holographic_amps:
                growth_amp = 0.5 * growth_amp + 0.5 * holographic_amps[0]
                decline_amp = 0.5 * decline_amp + 0.5 * holographic_amps[1]
                stagnation_amp = 0.5 * stagnation_amp + 0.5 * holographic_amps[2]
                fg_amp = 0.5 * fg_amp + 0.5 * holographic_amps[3]
                fd_amp = 0.5 * fd_amp + 0.5 * holographic_amps[4]
                fs_amp = 0.5 * fs_amp + 0.5 * holographic_amps[5]
            # Влияние ячейки
            if cell_amps:
                growth_amp = 0.8 * growth_amp + 0.2 * cell_amps[0]
                decline_amp = 0.8 * decline_amp + 0.2 * cell_amps[1]
                stagnation_amp = 0.8 * stagnation_amp + 0.2 * cell_amps[2]
                fg_amp = 0.8 * fg_amp + 0.2 * cell_amps[3]
                fd_amp = 0.8 * fd_amp + 0.2 * cell_amps[4]
                fs_amp = 0.8 * fs_amp + 0.2 * cell_amps[5]

        if price_change > 0:
            growth_amp += 0.4 if close_price == high_price else 0.3
            decline_amp += 0.1
            stagnation_amp += 0.2
            fg_amp += 0.2
        elif price_change < 0:
            decline_amp += 0.4 if close_price == low_price else 0.3
            growth_amp += 0.1
            stagnation_amp += 0.2
            fd_amp += 0.2
        else:
            stagnation_amp += 0.5
            growth_amp += 0.25
            decline_amp += 0.25
            fs_amp += 0.2

        if avg_range != 0 and avg_volume != 0:
            phase = sigmoid(avg_price_change / avg_range + (turnover / avg_volume - 1) + 0.01 * prev_error) * 2 * math.pi
        else:
            phase = 0
        interference_factor = 0.15 * cmath.exp(1j * phase)
        if avg_price_change > 0:
            growth_amp += interference_factor
            fg_amp += interference_factor * 0.5
        elif avg_price_change < 0:
            decline_amp += interference_factor
            fd_amp += interference_factor * 0.5

        entanglement_boost = abs(trend_strength) * (volume / avg_volume) * 0.2
        if trend_strength > 0:
            growth_amp += entanglement_boost * cmath.exp(1j * phase)
            decline_amp -= entanglement_boost / 2
            fg_amp += entanglement_boost * 0.5 * cmath.exp(1j * phase)
        elif trend_strength < 0:
            decline_amp += entanglement_boost * cmath.exp(1j * phase)
            growth_amp -= entanglement_boost / 2
            fd_amp += entanglement_boost * 0.5 * cmath.exp(1j * phase)

        if range_percent < 0.05 or range_price < (avg_range * 0.8):
            stagnation_amp += 0.3
            fs_amp += 0.15
        elif range_percent > 0.2 or range_price > (avg_range * 1.2):
            growth_amp += 0.15
            decline_amp += 0.15
            fg_amp += 0.075
            fd_amp += 0.075

        # Корреляторы с прошлыми состояниями
        correlator_sum = 0
        correlator_count = 0
        for k in range(start_idx, current_idx):
            if k in prev_states:
                correlator = compute_correlator(
                    (growth_amp, decline_amp, stagnation_amp, fg_amp, fd_amp, fs_amp),
                    prev_states[k]
                )
                correlator_sum += correlator
                correlator_count += 1
        avg_correlator = correlator_sum / correlator_count if correlator_count > 0 else 0.5
        ensemble_correlators.append(avg_correlator)
        
        # Усиление амплитуд пропорционально корреляции
        correlation_boost = 0.1 * avg_correlator
        growth_amp *= (1 + correlation_boost)
        decline_amp *= (1 + correlation_boost)
        stagnation_amp *= (1 + correlation_boost)
        fg_amp *= (1 + correlation_boost)
        fd_amp *= (1 + correlation_boost)
        fs_amp *= (1 + correlation_boost)

        amps = np.array([growth_amp, decline_amp, stagnation_amp, fg_amp, fd_amp, fs_amp], dtype=complex)
        norm = math.sqrt(sum(abs(x)**2 for x in amps))
        if norm > 0:
            amps /= norm
        ensemble_amps.append(amps)

    # Усреднение по ансамблю
    mean_amps = np.mean(ensemble_amps, axis=0)
    growth_amp, decline_amp, stagnation_amp, fg_amp, fd_amp, fs_amp = mean_amps
    
    mean_correlator = np.mean(ensemble_correlators) if ensemble_correlators else 0.5
    # Энтропия фон Неймана
    entropy = compute_entropy(ensemble_amps)
    entropy_factor = 1 - min(entropy / math.log(6), 1)

    growth_prob = abs(growth_amp)**2 + abs(fg_amp)**2 * 0.5
    decline_prob = abs(decline_amp)**2 + abs(fd_amp)**2 * 0.5
    stagnation_prob = abs(stagnation_amp)**2 + abs(fs_amp)**2 * 0.5
    total = growth_prob + decline_prob + stagnation_prob
    if total > 0:
        growth_prob /= total
        decline_prob /= total
        stagnation_prob /= total
    # Корректировка вероятностей с коррелятором
    growth_prob *= (1 + 0.1 * mean_correlator)
    decline_prob *= (1 + 0.1 * mean_correlator)
    stagnation_prob *= (1 + 0.1 * mean_correlator)
    total = growth_prob + decline_prob + stagnation_prob
    if total > 0:
        growth_prob /= total
        decline_prob /= total
        stagnation_prob /= total

    return (growth_amp, decline_amp, stagnation_amp, fg_amp, fd_amp, fs_amp), (growth_prob, decline_prob, stagnation_prob), avg_range, entropy_factor, mean_correlator

def predict_close_price(df, current_idx, prev_states=None, prev_error=0, timeframe=1, ensemble_size=10, entangle_range=3, ensemble_size_traj=5, cell_size=5):
    """Предсказывает closePrice."""
    lookback = max(5, int(60 / timeframe))
    row = df.iloc[current_idx]
    
    # Расчёт волновой функции
    amps, probs, avg_range, entropy_factor, mean_correlator = calculate_wave_function(
        df, current_idx, prev_states, prev_error, timeframe, lookback, entangle_range, ensemble_size_traj, cell_size
    )
    growth_amp, decline_amp, stagnation_amp, fg_amp, fd_amp, fs_amp = amps
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

    return expected_close, collapsed_mean, ci_lower, ci_upper, amps, mean_correlator

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
ensemble_size = 20  # Для коллапса
ensemble_size_traj = 5  # Для ансамбля траекторий
entangle_range = 3
cell_size = 5
expected_prices = []
collapsed_prices = []
ci_lowers = []
ci_uppers = []
actual_prices = df["closePrice"].tolist()[1:]
times = df["startTime"].tolist()[:-1]
correlators = []
prev_states = {}
prev_error = 0

for i in range(len(df) - 1):
    expected_close, collapsed_mean, ci_lower, ci_upper, amps, mean_correlator = predict_close_price(
        df, i, prev_states, prev_error, timeframe, ensemble_size=ensemble_size, entangle_range=entangle_range, ensemble_size_traj=ensemble_size_traj, cell_size=cell_size
    )
    expected_prices.append(expected_close)
    collapsed_prices.append(collapsed_mean)
    ci_lowers.append(ci_lower)
    ci_uppers.append(ci_upper)
    correlators.append(mean_correlator)
    prev_states[i] = amps
    prev_error = actual_prices[i] - collapsed_mean if i < len(actual_prices) else 0

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

# Построение графиков
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# График цен
ax1.plot(times, actual_prices, label="Реальный closePrice", color="blue", marker="o")
ax1.plot(times, expected_prices, label="Ожидаемый closePrice (квант)", color="orange", linestyle="--", marker="x")
ax1.plot(times, collapsed_prices, label="Ансамбль closePrice (квант)", color="green", linestyle="-.", marker="^")
ax1.fill_between(times, ci_lowers, ci_uppers, color="green", alpha=0.2, label="95% Доверительный интервал (с энтропией)")
ax1.set_ylabel("Цена (USD)")
ax1.set_title(f"Сравнение цен (timeframe={timeframe} мин, ensemble_size={ensemble_size}, traj={ensemble_size_traj})")
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
