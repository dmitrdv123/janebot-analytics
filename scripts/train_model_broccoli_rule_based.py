import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from datetime import datetime
from deap import base, creator, tools, algorithms

from utils_features import calculate_rsi
from utils import load_data

def run_baseline(df, fee_open, fee_close, params):
  realized_pnl = 0

  for i in range(len(df)):
    current_row = df.iloc[i]
    if not np.isnan(current_row['fundingRate']):  # Funding rate
      gross_profit = params['position_amount_open'] * (df['closePrice'].iloc[0] - current_row['closePrice']) / df['closePrice'].iloc[0]
      position_amount_cur = params['position_amount_open'] + gross_profit
      realized_pnl += position_amount_cur * current_row['fundingRate']

  gross_profit = params['position_amount_open'] * (df['closePrice'].iloc[0] - df['closePrice'].iloc[-1]) / df['closePrice'].iloc[0]
  position_amount_cur = params['position_amount_open'] + gross_profit

  realized_pnl -= params['position_amount_open'] * fee_open + position_amount_cur * fee_close
  profit = (gross_profit + realized_pnl) / params['position_amount_open']

  return profit

def run_bot(df, amount, fee_open, fee_close, params, start_idx=0, end_idx=None):
  if end_idx is None:
    end_idx = len(df)

  profits = []  # Track profit per trade
  positions = []

  position_open = False
  position_open_timestamp = None
  total_amount = amount
  realized_pnl = 0

  position_open_timestamps = []
  position_close_timestamps = []
  position_prices = []
  position_amounts = []

  for i in range(start_idx, end_idx):
    current_row = df.iloc[i]
    price_close_cur = current_row['closePrice']
    start_time = current_row['startTime']
    rsi_1h = current_row['rsi_1h']
    zscore_1h = current_row['zscore_1h']
    funding_rate = current_row['fundingRate']

    if not position_open and zscore_1h > params['zscore_threshold_open'] and rsi_1h > params['rsi_threshold_open']:  # Open short
      position_amount_open = min(amount * params['position_amount_open'], total_amount)
      if position_amount_open > 0:
        position_amounts.append(position_amount_open)
        position_prices.append(price_close_cur)
        positions.append(1)
        position_open_timestamps.append(start_time)

        fee_open_amount = position_amount_open * fee_open
        realized_pnl = -fee_open_amount
        total_amount -= position_amount_open + fee_open_amount
        position_open_timestamp = start_time
        position_open = True
    elif position_open:
      gross_profit = 0
      position_amount_sum = 0
      for j in range(len(position_amounts)):
        position_amount_sum += position_amounts[j]
        gross_profit += position_amounts[j] * (position_prices[j] - price_close_cur) / position_prices[j]
      position_amount_cur = position_amount_sum + gross_profit

      if not np.isnan(funding_rate):  # Funding rate
        funding_rate_amount = position_amount_cur * funding_rate
        realized_pnl += funding_rate_amount
        total_amount += funding_rate_amount

      # Close position if position duration over threshold or last record
      duration = (start_time - position_open_timestamp).total_seconds() / 3600  # Convert to hours
      if duration > params['max_duration_hours'] or i == end_idx - 1:  # Close position if duration exceeds max_duration_hours
        fee_close_amount = position_amount_cur * fee_close
        realized_pnl -= fee_close_amount
        total_amount += position_amount_cur - fee_close_amount

        profit = (gross_profit + realized_pnl) / position_amount_sum
        profits.append(profit)
        position_close_timestamps.append(start_time)

        realized_pnl = 0
        position_open_timestamp = None
        position_prices = []
        position_amounts = []
        position_open = False

        continue

      # Close position if rsi below threshold and profit over threshold
      if rsi_1h < params['rsi_threshold_close']:
        fee_close_amount = position_amount_cur * fee_close
        profit = (gross_profit + realized_pnl - fee_close_amount) / position_amount_sum
        if profit > params['profit_min']:
          realized_pnl -= fee_close_amount
          total_amount += position_amount_cur - fee_close_amount

          profits.append(profit)
          position_close_timestamps.append(start_time)

          realized_pnl = 0
          position_open_timestamp = None
          position_prices = []
          position_amounts = []
          position_open = False

          continue

      # Increase position if necessary
      zscore_1h = current_row['zscore_1h']
      closePrice_pct_change_since_open = (price_close_cur - position_prices[-1]) / price_close_cur
      position_amount_increase = min(amount * params['position_amount_increase'], total_amount)
      if position_amount_increase > 0 and closePrice_pct_change_since_open > params['price_diff_threshold_increase'] and rsi_1h > params['rsi_threshold_increase']:  # Increase short
        position_amounts.append(position_amount_increase)
        position_prices.append(price_close_cur)
        positions[-1] += 1
        position_open_timestamps.append(start_time)

        fee_open_amount = position_amount_increase * fee_open
        realized_pnl -= fee_open_amount
        total_amount -= position_amount_increase + fee_open_amount

        continue

  return total_amount, profits, positions, position_open_timestamps, position_close_timestamps

# Определяем фитнес-функцию (максимизация прибыли)
def evaluate(individual, df, amount, fee_open, fee_close):
  params = {
    'position_amount_open': individual[0],
    'position_amount_increase': individual[1],
    'profit_min': individual[2],
    'zscore_threshold_open': individual[3],
    'price_diff_threshold_increase': individual[4],
    'rsi_threshold_open': individual[5],
    'rsi_threshold_increase': individual[6],
    'rsi_threshold_close': individual[7],
    'max_duration_hours': individual[8],
  }
  total_amount, profits, positions, _, _ = run_bot(df, amount, fee_open, fee_close, params)
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

def mutate_with_bounds(individual, mu, sigma, indpb, low, up):
  tools.mutGaussian(individual, mu=mu, sigma=sigma, indpb=indpb)
  for i in range(len(individual)):
    individual[i] = min(max(individual[i], low[i]), up[i])
  return individual,

def find_optimal_params_ga(df_data, amount, fee_open, fee_close, param_ranges):
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
  toolbox.register("evaluate", evaluate, df=df_data, amount=amount, fee_open=fee_open, fee_close=fee_close)
  toolbox.register("mate", tools.cxBlend, alpha=0.5)
  toolbox.register("mutate", mutate_with_bounds, mu=0, sigma=0.1, indpb=0.2, low=[r[0] for r in param_ranges], up=[r[1] for r in param_ranges])
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
    'zscore_threshold_open': best_individual[3],
    'price_diff_threshold_increase': best_individual[4],
    'rsi_threshold_open': best_individual[5],
    'rsi_threshold_increase': best_individual[6],
    'rsi_threshold_close': best_individual[7],
    'max_duration_hours': best_individual[8],
  }
  best_reward = evaluate(best_individual, df_data, amount, fee_open, fee_close)[0]

  return best_reward, best_params

if __name__ == '__main__':
  symbol = 'BROCCOLIUSDT'
  # symbol = 'MELANIAUSDT'
  # symbol = 'TRUMPUSDT'
  # symbol = 'FARTCOINUSDT'
  # symbol = 'BTCUSDT'
  interval = '5'  # Kline interval (1m, 5m, 15m, etc.)
  time_start = '2024-01-01'
  amount = 10000

  fee_open = 0.0002
  fee_close = 0.0002

  param_ranges = [
    (0.05, 0.5),   # position_amount_open
    (0.01, 0.25),  # position_amount_increase
    (0.01, 0.5),   # profit_min
    (1, 3),        # zscore_threshold_open
    (0.005, 0.5),  # price_diff_threshold_increase
    (70, 90),      # rsi_threshold_open
    (70, 90),      # rsi_threshold_increase
    (10, 30),      # rsi_threshold_close
    (24, 2 * 24),  # max_duration_hours
  ]

  params = {
    'position_amount_open': 0.1,
    'position_amount_increase': 0.05,
    'profit_min': 0.1,
    'zscore_threshold_open': 1.5,
    'price_diff_threshold_increase': 0.05,
    'rsi_threshold_open': 80,
    'rsi_threshold_increase': 80,
    'rsi_threshold_close': 20,
    'max_duration_hours': 5 * 24,
  }

  # Load data
  df_data_kline = load_data(f'data/kline/{symbol}/{interval}')
  df_data_funding_rate = load_data(f'data/funding_rate/{symbol}')

  df_data_kline['startTime'] = pd.to_datetime(df_data_kline['startTime'], unit='ms')
  df_data_funding_rate['fundingRateTimestamp'] = pd.to_datetime(df_data_funding_rate['fundingRateTimestamp'], unit='ms')

  # Merge data
  df_data = pd.merge(df_data_kline, df_data_funding_rate, how='left', left_on='startTime', right_on='fundingRateTimestamp')
  df_data = df_data[df_data['startTime'] > pd.to_datetime(time_start)]

  # Calculate features
  df_data['closePrice_diff_1h'] = df_data['closePrice'].pct_change(periods=12)  # 1 hour = 12 periods
  df_data['rsi_1h'] = calculate_rsi(df_data, period=12)  # 1 hour = 12 periods
  df_data['zscore_1h'] = (df_data['closePrice'] - df_data['closePrice'].rolling(12).mean()) / df_data['closePrice'].rolling(12).std()

  # Drop records without features
  df_data = df_data.iloc[12:].reset_index(drop=True)

  # Find optimal params
  best_reward, best_params = find_optimal_params_ga(df_data, amount, fee_open, fee_close, param_ranges)
  print(f'Best rewards: {best_reward:.4f}')
  print(f'Best params: {best_params}')

  # Final run with optimal params
  params = best_params
  profit_baseline = run_baseline(df_data, fee_open, fee_close, params)
  total_amount, profits, positions, position_open_timestamps, position_close_timestamps = run_bot(df_data, amount, fee_open, fee_close, params)

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

  expected_value = 0.0
  profit_factor = 0.0
  reward_to_risk_ratio = 0.0
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

  print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: \nResult')

  print(f'Positions Count: {len(positions)}')
  print(f'Positions Increasing: {positions}')
  print(f'Max Position Increasing: {max(positions) if positions else 0}')
  print(f'Profits: {profits}')

  print(f'Position Durations: {[str(pd.to_timedelta(duration, unit="s")) for duration in position_durations]}')
  print(f'Max Position Duration: {str(pd.to_timedelta(max(position_durations, default=pd.Timedelta(0)).total_seconds(), unit="s"))}')

  print(f'Baseline Total Profit/Loss: {100 * profit_baseline:.2f} %')
  print(f'Total Profit/Loss: {100 * total_profit:.2f} %')

  print(f'Total Trade Profit/Loss: {100 * total_trade_profit:.2f} %')
  print(f'Avg Profit per Trade: {100 * avg_profit_per_trade:.2f} %')

  print(f'Profit Factor: {profit_factor:.4f}')
  print(f'Expected Value (EV): {100 * expected_value:.4f} %')
  print(f'Reward to Risk Ration (RRR): {100 * reward_to_risk_ratio:.4f} %')

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
