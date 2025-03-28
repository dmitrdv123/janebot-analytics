import random
import joblib
import numpy as np
import pandas as pd
import glob
import os
from deap import base, creator, tools, algorithms
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp

from utils_broccoli import run_bot, run_baseline, run_bot_simple, run_bot_simple_v2, run_bot_v3, run_random_strategy, scale_features, unscale_features
from utils import load_data
from constants import columns_input_kline

# Определяем фитнес-функцию (максимизация прибыли)
def evaluate(individual, df, y, amount, fee_open, fee_close, params):
  params = {
    'position_amount_open': individual[0],
    'position_amount_increase': individual[1],
    'profit_min': individual[2],
    'price_diff_threshold_open': individual[3],
    'price_diff_threshold_increase': individual[4],
    'rsi_threshold_open': individual[5],
    'rsi_threshold_increase': individual[6],
    'rsi_threshold_close': individual[7],
    'max_duration_hours': individual[8],
  }

  total_amount, profits, positions, _, _ = run_bot_v3(df, y, amount, fee_open, fee_close, params)
  total_profit = (total_amount - amount) / amount

  # Подсчитываем метрики
  n_winning_trades = len([p for p in profits if p > 0])  # Количество прибыльных сделок
  n_losing_trades = len([p for p in profits if p < 0])   # Количество убыточных сделок
  total_trades = len(profits) if profits else 1           # Общее количество сделок
  n_positions = len(positions) if positions else 1        # Общее количество позиций

  # Веса для компонентов вознаграждения
  w1 = 1.0  # Вес для общей прибыли
  w2 = 0.5  # Вес для доли выигрышных сделок
  w3 = 0.5  # Вес штрафа за убыточные сделки
  w4 = 1  # Вес для количества позиций

  # Нормализующий фактор для позиций (можно настроить)
  max_positions = len(df) / 10  # Пример: ожидаем до 10% точек данных как позиции

  # Формула вознаграждения
  reward = (
    w1 * total_profit +
    w2 * n_winning_trades / (n_winning_trades + n_losing_trades + 1e-6) -
    w3 * n_losing_trades / total_trades +
    w4 * n_positions / max_positions
  )

  return (reward,)  # Возвращаем кортеж для deap

def find_optimal_params_ga(df, y, amount, fee_open, fee_close, params, param_ranges):
  # Очистка предыдущих определений creator, если они есть
  if "FitnessMax" in creator.__dict__:
    del creator.FitnessMax
  if "Individual" in creator.__dict__:
    del creator.Individual

  # Настраиваем генетический алгоритм
  creator.create("FitnessMax", base.Fitness, weights=(1.0,))
  creator.create("Individual", list, fitness=creator.FitnessMax)

  toolbox = base.Toolbox()

  for i, (min_val, max_val) in enumerate(param_ranges):
    toolbox.register(f"attr_{i}", random.uniform, min_val, max_val)

  toolbox.register("individual", tools.initCycle, creator.Individual, [toolbox.attr_0, toolbox.attr_1, toolbox.attr_2, toolbox.attr_3, toolbox.attr_4, toolbox.attr_5, toolbox.attr_6, toolbox.attr_7, toolbox.attr_8], n=1)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  toolbox.register("evaluate", evaluate, df=df, y=y, amount=amount, fee_open=fee_open, fee_close=fee_close, params=params)
  toolbox.register("mate", tools.cxBlend, alpha=0.5)
  toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=[r[0] for r in param_ranges], up=[r[1] for r in param_ranges], indpb=0.2)
  toolbox.register("select", tools.selTournament, tournsize=3)

  # Запускаем алгоритм
  population = toolbox.population(n=50)  # Размер популяции
  ngen = 20                              # Количество поколений
  cxpb = 0.5                             # Вероятность скрещивания
  mutpb = 0.2                            # Вероятность мутации

  # Add statistics
  stats = tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("avg", np.mean)
  stats.register("max", np.max)

  pop, log = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=stats, verbose=True)

  # Извлекаем лучшее решение
  best_individual = tools.selBest(pop, k=1)[0]
  best_params = {
    'position_amount_open': best_individual[0],
    'position_amount_increase': best_individual[1],
    'profit_min': best_individual[2],
    'price_diff_threshold_open': best_individual[3],
    'price_diff_threshold_increase': best_individual[4],
    'rsi_threshold_open': best_individual[5],
    'rsi_threshold_increase': best_individual[6],
    'rsi_threshold_close': best_individual[7],
    'max_duration_hours': best_individual[8],
  }
  best_reward = evaluate(best_individual, df, y, amount, fee_open, fee_close, params)[0]

  return best_reward, best_params

if __name__ == '__main__':
  # symbol = 'BROCCOLIUSDT'
  # symbol = 'MELANIAUSDT'
  # symbol = 'TRUMPUSDT'
  symbol = 'FARTCOINUSDT'
  # symbol = 'BTCUSDT'
  interval = '5'  # Kline interval (1m, 5m, 15m, etc.)
  time_start = '2024-01-01'
  amount = 10000
  timesteps = 5
  model_name = 'lstm'

  fee_open = 0.0002
  fee_close = 0.0002

  # Define feature columns
  columns_input = columns_input_kline

  param_ranges = [
    (0.05, 0.5),   # position_amount_open
    (0.01, 0.25),  # position_amount_increase
    (0.01, 0.5),   # profit_min
    (0.01, 0.5),   # price_diff_threshold_open
    (0.005, 0.5),  # price_diff_threshold_increase
    (70, 90),      # rsi_threshold_open
    (70, 90),      # rsi_threshold_increase
    (10, 30),      # rsi_threshold_close
    (24, 2 * 24),  # max_duration_hours
  ]

  params_init = {
    'position_amount_open': 0.1,
    'position_amount_increase': 0.05,
    'profit_min': 0.1,
    'price_diff_threshold_open': 0.1,
    'price_diff_threshold_increase': 0.05,
    'rsi_threshold_open': 80,
    'rsi_threshold_increase': 80,
    'rsi_threshold_close': 20,
    'max_duration_hours': 5 * 24,
  }

  dir = f'models/{symbol}/{model_name}'

  # Load the model
  model = load_model(f'{dir}/model.keras')

  # Load scalers
  scalers = {}
  for file_path in glob.glob(f'{dir}/scaler_*.pkl'):
    scaler_name = os.path.basename(file_path).replace(f'scaler_', '').replace('.pkl', '')
    scalers[scaler_name] = joblib.load(file_path)

  # Load data
  df_features_kline = pd.read_csv(f'features/kline/{symbol}/{interval}/features.csv')
  df_data_funding_rate = load_data(f'data/funding_rate/{symbol}')

  # Take the last 20% of df_features_kline
  df_features_kline = df_features_kline.tail(int(len(df_features_kline) * 0.2))

  df_features_kline['startTime'] = pd.to_datetime(df_features_kline['startTime'], unit='ms')
  df_data_funding_rate['fundingRateTimestamp'] = pd.to_datetime(df_data_funding_rate['fundingRateTimestamp'], unit='ms')

  # Scale features
  df_features_kline_scaled, _ = scale_features(df_features_kline, scalers)

  # Merge data
  df_data = pd.merge(df_features_kline_scaled, df_data_funding_rate, how='left', left_on='startTime', right_on='fundingRateTimestamp')

  # Drop duplicates by startTime
  df_data = df_data.drop_duplicates(subset='startTime')

  # Filter data
  df_data = df_data[df_data['startTime'] > pd.to_datetime(time_start)]

  # Order by timestamp ascending
  df_data = df_data.sort_values(by='startTime')

  # Prepare input data
  X = df_data[columns_input].values

  # Create sequences of 5 timesteps
  X_seq = []
  for i in range(len(X)):
    if i < timesteps - 1:
      # For the first few rows, pad with the earliest available data
      start_idx = 0
    else:
      start_idx = i - (timesteps - 1)
    X_seq.append(X[start_idx:i+1])  # Take the last 5 timesteps (or less for early rows)

  # Pad sequences to ensure they all have 5 timesteps
  for i in range(len(X_seq)):
    if len(X_seq[i]) < timesteps:
      # Pad with the first row repeated
      padding = np.repeat(X_seq[i][:1], timesteps - len(X_seq[i]), axis=0)
      X_seq[i] = np.vstack((padding, X_seq[i]))
  X_seq = np.array(X_seq)  # Shape: (samples, timesteps, features)

  # Convert to float32 for TensorFlow
  X_seq = X_seq.astype(np.float32)

  # Make predictions
  y = model.predict(X_seq)

  df_data = unscale_features(df_data, scalers)
  y = scalers['output'].inverse_transform(y.reshape(-1, 1))

  # # Find optimal params
  # reward_best, params_best = find_optimal_params_ga(df_data, y, amount, fee_open, fee_close, params_init, param_ranges)
  # print(f'Best rewards: {reward_best:.4f}')
  # print(f'Best params: {params_best}')

  # Final run with optimal params
  params = {
    "position_amount_open": 0.5,
    "position_amount_increase": 0.1,
    "profit_min": 0.015,
    "price_diff_threshold_open": 0.06,
    "price_diff_threshold_increase": 0.01,
    "rsi_threshold_open": 80,
    "rsi_threshold_increase": 80,
    "rsi_threshold_close": 20,
    "max_duration_hours": 48,
    
    "price_diff_future_threshold_open": 0.1
  }
  
  # Add to your main script
  random_amount, random_profits, random_positions, random_open_timestamps, random_close_timestamps = run_random_strategy(df_data, amount, fee_open, fee_close, params)
  random_profit = (random_amount - amount) / amount

  profit_baseline = run_baseline(df_data, fee_open, fee_close, params)

  total_amount, profits, positions, position_open_timestamps, position_close_timestamps = run_bot_simple_v2(df_data, y, amount, fee_open, fee_close, params)

  # Calculate position durations
  position_durations = []
  previous_close_time = None
  for close_time in position_close_timestamps:
    open_times_in_range = [
      open_time for open_time in position_open_timestamps 
      if open_time <= close_time and (previous_close_time is None or open_time >= previous_close_time)
    ]
    if open_times_in_range:
      position_durations.append((close_time - min(open_times_in_range)))
    previous_close_time = close_time

  total_profit = (total_amount - amount) / amount
  total_trade_profit = sum(profits) if profits else 0.0
  avg_profit_per_trade = total_trade_profit / len(positions) if positions else 0
  
  # Calculate mean and standard deviation of profits
  profits_array = np.array(profits)
  std_profit = np.std(profits_array, ddof=1)  # ddof=1 for sample standard deviation
  n_trades = len(profits_array)

  expected_value = 0.0
  profit_factor = 0.0
  if profits:
    total_gains = sum(p for p in profits if p > 0)
    total_losses = abs(sum(p for p in profits if p < 0))
    profit_factor = total_gains / total_losses if total_losses != 0 else (total_gains if total_gains > 0 else 0.0)

    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p < 0]

    avg_profit = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0.0
    p_win = len(winning_trades) / len(profits) if profits else 0.0
    p_loss = 1 - p_win

    expected_value = (p_win * avg_profit) - (p_loss * abs(avg_loss))
    reward_to_risk_ratio = avg_profit / abs(avg_loss) if avg_loss != 0 else float('inf') if avg_profit > 0 else 0.0

  print('\nResult')

  print(f'Positions Count: {len(positions)}')
  print(f'Positions Increasing: {positions}')
  print(f'Max Position Increasing: {max(positions) if positions else 0}')

  print(f'Position Durations: {[str(pd.to_timedelta(duration, unit="s")) for duration in position_durations]}')
  print(f'Max Position Duration: {str(pd.to_timedelta(max(position_durations, default=pd.Timedelta(0)).total_seconds(), unit="s"))}')

  print(f'Random Strategy Total Profit/Loss: {100 * random_profit:.2f} %')
  print(f'Random Strategy Positions Count: {len(random_positions)}')
  print(f'Baseline Total Profit/Loss: {100 * profit_baseline:.2f} %')
  print(f'Total Profit/Loss: {100 * total_profit:.2f} %')

  print(f'Total Trade Profit/Loss: {100 * total_trade_profit:.2f} %')
  print(f'Avg Profit per Trade: {100 * avg_profit_per_trade:.2f} %')

  print(f'Count of Negative Profits: {len([p for p in profits if p <= 0])}')
  print(f'Count of Positive Profits: {len([p for p in profits if p > 0])}')
  print(f'Profit Factor: {profit_factor:.4f}')
  print(f'Expected Value (EV): {100 * expected_value:.4f} %')
  print(f'Reward to Risk Ration (RRR): {100 * reward_to_risk_ratio:.4f} %')

  print(f'Standard Deviation of Profits: {100 * std_profit:.2f} %')

  # Perform a one-sample t-test to check if the mean profit is significantly different from 0
  t_stat, p_value = ttest_1samp(profits_array, 0)
  print(f'T-test p-value: {p_value:.4f}')
  if p_value < 0.05:
      print("The mean profit is significantly different from 0 (p < 0.05).")
  else:
      print("The mean profit is not significantly different from 0 (p >= 0.05).")

  # Plot closePrice over time
  plt.figure(figsize=(12, 6))
  plt.plot(df_data['startTime'], df_data['closePrice'], label='Close Price', color='blue')

  # Plot position open timestamps
  open_prices = [df_data[df_data['startTime'] == ts]['closePrice'].values[0] for ts in position_open_timestamps]
  plt.scatter(position_open_timestamps, open_prices, color='green', label='Position Open', marker='o')

  # Plot position close timestamps
  close_prices = [df_data[df_data['startTime'] == ts]['closePrice'].values[0] for ts in position_close_timestamps]
  plt.scatter(position_close_timestamps, close_prices, color='red', label='Position Close', marker='x')

  # Add labels and legend
  plt.xlabel('Timestamp')
  plt.ylabel('Close Price')
  plt.title('Close Price with Position Open/Close Points')
  plt.legend()
  plt.grid()

  # Show the plot
  plt.show()
