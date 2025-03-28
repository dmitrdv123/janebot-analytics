import json
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from datetime import datetime

from utils_features import calculate_rsi
from utils import load_data

# Parameter ranges
param_ranges = {
  'position_amount_open': [0.05, 0.5],
  'position_amount_increase': [0.01, 0.25],
  'profit_min': [0.01, 0.5],
  'price_diff_threshold_open': [0.01, 0.5],
  'price_diff_threshold_increase': [0.005, 0.5],
  'rsi_threshold_open': [70, 90],
  'rsi_threshold_increase': [70, 90],
  'rsi_threshold_close': [10, 30],
  'max_duration_hours': [12, 24 * 7]
}

class DQNAgent:
  def __init__(self, state_size, action_size, param_ranges):
    self.state_size = state_size
    self.action_size = action_size
    self.param_ranges = param_ranges
    self.memory = deque(maxlen=2000)
    self.gamma = 0.95  # Discount factor
    self.epsilon = 1.0  # Exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.95
    self.model = self._build_model()

  def _build_model(self):
    model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(self.state_size,)),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(self.action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

  def remember(self, state, actions, reward, next_state, done):
    self.memory.append((state, actions, reward, next_state, done))

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return np.random.randint(0, 3, size=len(self.param_ranges))  # Random actions for each parameter
    act_values = self.model.predict(state, verbose=0)[0]
    return [int(act_values[i * 3:(i + 1) * 3].argmax()) for i in range(len(self.param_ranges))]

  def replay(self, batch_size):
    if len(self.memory) < batch_size:
      return
    minibatch = random.sample(self.memory, batch_size)
    states = np.array([m[0][0] for m in minibatch])  # Shape: (batch_size, state_size)
    next_states = np.array([m[3][0] for m in minibatch])  # Shape: (batch_size, state_size)
    targets = self.model.predict(states, verbose=0)  # Batch prediction
    next_q_values = self.model.predict(next_states, verbose=0)  # Batch prediction

    for i, (state, actions, reward, next_state, done) in enumerate(minibatch):
      target = reward
      if not done:
        target = reward + self.gamma * np.amax(next_q_values[i])
      for j, action in enumerate(actions):
        targets[i][j * 3 + action] = target

    self.model.fit(states, targets, epochs=1, verbose=0)

# State extraction
def get_state(df, index, params):
  rsi_1h = df['rsi_1h'].iloc[index]
  close_price_diff_1h = df['closePrice_diff_1h'].iloc[index]
  funding_rate = df['fundingRate'].iloc[index] if not np.isnan(df['fundingRate'].iloc[index]) else 0
  volume = df['volume'].iloc[index]
  turnover = df['turnover'].iloc[index]
  param_values = [params[key] for key in sorted(params.keys())]
  return np.array([[rsi_1h, close_price_diff_1h, funding_rate, volume, turnover] + param_values])

# Apply action to parameter
def apply_actions(params, actions, step_size=0.02):
  for _, (param_name, action) in enumerate(zip(sorted(params.keys()), actions)):
    min_val, max_val = param_ranges[param_name]
    step = step_size if 'rsi' not in param_name else 2
    current_value = params[param_name]

    if action == 0:  # Decrease
      new_value = current_value - step
    elif action == 2:  # Increase
      new_value = current_value + step
    else:  # Keep (action == 1)
      new_value = current_value

    # Clamp to bounds
    params[param_name] = max(min_val, min(max_val, new_value))

  return params

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
    close_price_diff_1h = current_row['closePrice_diff_1h']
    funding_rate = current_row['fundingRate']

    if not position_open and close_price_diff_1h > params['price_diff_threshold_open'] and rsi_1h > params['rsi_threshold_open']:  # Open short
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

if __name__ == '__main__':
  symbol = 'BROCCOLIUSDT'
  # symbol = 'MELANIAUSDT'
  # symbol = 'TRUMPUSDT'
  # symbol = 'FARTCOINUSDT'
  # symbol = 'BTCUSDT'
  interval = '5'  # Kline interval (1m, 5m, 15m, etc.)
  time_start = '2025-02-21'
  amount = 10000

  fee_open = 0.0002
  fee_close = 0.0002

  params_manual = {
    'position_amount_open': 0.1,
    'position_amount_increase': 0.05,
    'profit_min': 0.1,
    'price_diff_threshold_open': 0.1,
    'price_diff_threshold_increase': 0.05,
    'rsi_threshold_open': 80,
    'rsi_threshold_increase': 80,
    'rsi_threshold_close': 20,
    'max_duration_hours': 7 * 24,
  }

  # Load data
  df_data_kline = load_data(f'data/kline/{symbol}/{interval}')
  df_data_funding_rate = load_data(f'data/funding_rate/{symbol}')

  df_data_kline['startTime'] = pd.to_datetime(df_data_kline['startTime'], unit='ms')
  df_data_funding_rate['fundingRateTimestamp'] = pd.to_datetime(df_data_funding_rate['fundingRateTimestamp'], unit='ms')

  # Merge data
  df_data = pd.merge(df_data_kline, df_data_funding_rate, how='left', left_on='startTime', right_on='fundingRateTimestamp')
  df_data = df_data[df_data['startTime'] > pd.to_datetime(time_start)]

  # Calculate metrics
  df_data['closePrice_diff_1h'] = df_data['closePrice'].pct_change(periods=12)  # 1 hour = 12 periods
  df_data['rsi_1h'] = calculate_rsi(df_data, period=12)  # 1 hour = 12 periods

  # Initialize DQN
  state_size = 5 + len(param_ranges)  # 5 market features + 9 params = 14
  action_size = 3 * len(param_ranges)  # 3 actions per parameter * 9 params = 27
  agent = DQNAgent(state_size, action_size, param_ranges)
  batch_size = 32
  episodes = 50
  window_size = 576  # Window for evaluation - 2 days (48 hours * 12 periods/hour)
  step_size = 12  # Step every hour

  # Training DQN
  params = params_manual.copy()
  episode_results = []
  for e in range(episodes):
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Episode {e + 1} / {episodes}')

    total_reward = 0
    current_params = params.copy()
    for i in range(12, len(df_data) - window_size, step_size):
      state = get_state(df_data, i, current_params)
      actions = agent.act(state)
      current_params = apply_actions(current_params, actions)
      total_amount, profits, positions, _, _ = run_bot(df_data, amount, fee_open, fee_close, current_params, i, i + window_size)

      if profits:
        total_gains = sum(p for p in profits if p > 0) if profits else 0.0
        total_losses = abs(sum(p for p in profits if p < 0)) if profits else 0.0
        profit_factor = total_gains / total_losses if total_losses != 0 else (total_gains if total_gains > 0 else 0.0)
        reward = min(profit_factor, 5.0) + 1.0 * len(positions) + 10 * np.mean(profits)
      else:
        reward = -2.0

      total_reward += reward
      next_state = get_state(df_data, i + 1, current_params)
      done = i == len(df_data) - window_size - step_size
      agent.remember(state, actions, reward, next_state, done)

    if len(agent.memory) > batch_size:
      agent.replay(batch_size)
    if agent.epsilon > agent.epsilon_min:
      agent.epsilon *= agent.epsilon_decay

    # Full run to evaluate episode
    total_amount, profits, positions, _, _ = run_bot(df_data, amount, fee_open, fee_close, current_params)
    final_profit = (total_amount - amount) / amount
    total_gains = sum(p for p in profits if p > 0) if profits else 0.0
    total_losses = abs(sum(p for p in profits if p < 0)) if profits else 0.0
    profit_factor = total_gains / total_losses if total_losses != 0 else (total_gains if total_gains > 0 else 0.0)
    
    episode_results.append({
      'episode': e + 1,
      'total_reward': total_reward,
      'final_profit': final_profit,
      'profit_factor': min(profit_factor, 10.0),
      'num_trades': len(positions),
      'params': current_params.copy()
    })

    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Total Reward: {total_reward:.4f}, Final Profit: {100 * final_profit:.2f}%, Profit Factor: {min(profit_factor, 10.0):.4f}, Trades: {len(positions)}, Epsilon: {agent.epsilon:.4f}')

    params = current_params

  # Save model and select best params
  agent.model.save(f'models/model_broccoli.keras')
  best_result = max([r for r in episode_results if r['profit_factor'] > 1.5 and r['num_trades'] >= 5], 
           key=lambda x: x['final_profit'], default=episode_results[-1])
  optimal_params = best_result['params']
  with open('models/model_broccoli_optimal_params.json', 'w') as f:
    json.dump(optimal_params, f)
  print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Model saved')
  print(f'Optimal params from episode {best_result["episode"]} with profit {100 * best_result["final_profit"]:.2f}%, Profit Factor {best_result["profit_factor"]:.4f}, Trades {best_result["num_trades"]}')

  # Run manual params for comparison
  manual_amount, manual_profits, manual_positions, _, _ = run_bot(df_data, amount, fee_open, fee_close, params_manual)
  manual_profit = (manual_amount - amount) / amount
  manual_gains = sum(p for p in manual_profits if p > 0) if manual_profits else 0.0
  manual_losses = abs(sum(p for p in manual_profits if p < 0)) if manual_profits else 0.0
  manual_profit_factor = manual_gains / manual_losses if manual_losses != 0 else (manual_gains if manual_gains > 0 else 0.0)
  print(f'Manual params profit: {100 * manual_profit:.2f}%, Profit Factor: {min(manual_profit_factor, 10.0):.4f}, Trades: {len(manual_positions)}')

  # Final run with optimal params
  params = optimal_params
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

  print(f'Total Amount: {total_amount}')
  print(f'Position Total: {len(positions)}')
  print(f'Position Increasing: {positions}')
  print(f'Max Position Increasing: {max(positions) if positions else 0}')

  print(f'Position Durations: {[str(pd.to_timedelta(duration, unit="s")) for duration in position_durations]}')
  print(f'Max Position Duration: {str(pd.to_timedelta(max(position_durations, default=pd.Timedelta(0)).total_seconds(), unit="s"))}')

  print(f'Baseline total profit/loss: {100 * profit_baseline:.2f} %')
  print(f'Total profit/loss: {100 * total_profit:.2f} %')

  print(f'Total trade profit/loss: {100 * total_trade_profit:.2f} %')
  print(f'Avg profit per trade: {100 * avg_profit_per_trade:.2f} %')

  print(f'Expected Value (EV): {100 * expected_value:.4f} %')
  print(f'Profit Factor: {profit_factor:.4f}')

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
