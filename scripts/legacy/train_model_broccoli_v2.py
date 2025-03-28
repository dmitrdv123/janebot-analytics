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

class DQNAgent:
  def __init__(self, state_size, action_size=4):
    self.state_size = state_size
    self.action_size = action_size  # 4 actions: Open, Increase, Hold, Close
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

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return np.random.randint(self.action_size)  # Random action: 0, 1, 2, or 3
    act_values = self.model.predict(state, verbose=0)[0]
    return np.argmax(act_values)  # Best action

  def replay(self, batch_size):
    if len(self.memory) < batch_size:
      return
    minibatch = random.sample(self.memory, batch_size)
    states = np.array([m[0][0] for m in minibatch])
    next_states = np.array([m[3][0] for m in minibatch])
    targets = self.model.predict(states, verbose=0)
    next_q_values = self.model.predict(next_states, verbose=0)
    for i, (state, action, reward, next_state, done) in enumerate(minibatch):
      target = reward if done else reward + self.gamma * np.max(next_q_values[i])
      targets[i][action] = target
    self.model.fit(states, targets, epochs=1, verbose=0)

def get_state(df, index):
  rsi_1h = df['rsi_1h'].iloc[index]
  close_price_diff_1h = df['closePrice_diff_1h'].iloc[index]
  funding_rate = df['fundingRate'].iloc[index] if not np.isnan(df['fundingRate'].iloc[index]) else 0
  volume = df['volume'].iloc[index]
  turnover = df['turnover'].iloc[index]
  return np.array([[rsi_1h, close_price_diff_1h, funding_rate, volume, turnover]])

def run_bot(df, amount, fee_open, fee_close, agent, train=True, episodes=1, start_idx=12, step_size=1, max_duration_hours=168):
  open_amount = 1000  # Fixed amount for opening a position
  increase_amount = 500  # Fixed amount for increasing a position
  batch_size = 32
  total_profits = []
  all_positions = []
  all_open_timestamps = []
  all_close_timestamps = []

  for episode in range(episodes):
    profits = []
    positions = []
    position_open = False
    position_open_timestamp = None
    total_amount = amount
    realized_pnl = 0
    position_prices = []
    position_amounts = []
    position_open_timestamps = []
    position_close_timestamps = []
    no_trade_steps = 0

    for i in range(start_idx, len(df), step_size):
      state = get_state(df, i)
      action = agent.act(state)
      current_row = df.iloc[i]
      price_close_cur = current_row['closePrice']
      start_time = current_row['startTime']
      funding_rate = current_row['fundingRate']

      if not position_open:
        if action == 0 and total_amount >= open_amount:  # Open
          position_amounts.append(open_amount)
          position_prices.append(price_close_cur)
          positions.append(1)
          position_open_timestamps.append(start_time)
          fee_open_amount = open_amount * fee_open
          realized_pnl = -fee_open_amount
          total_amount -= open_amount + fee_open_amount
          position_open = True
          position_open_timestamp = start_time
          no_trade_steps = 0
          reward = 0
        else:
          no_trade_steps += 1
          reward = -0.2 * no_trade_steps if no_trade_steps > 5 else 0  # Escalating penalty for inaction
      else:
        # Calculate current position value
        gross_profit = 0
        position_amount_sum = sum(position_amounts)
        for j in range(len(position_amounts)):
          gross_profit += position_amounts[j] * (position_prices[j] - price_close_cur) / position_prices[j]
        position_amount_cur = position_amount_sum + gross_profit

        if not np.isnan(funding_rate):
          funding_rate_amount = position_amount_cur * funding_rate
          realized_pnl += funding_rate_amount
          total_amount += funding_rate_amount

        reward = 0
        duration = (start_time - position_open_timestamp).total_seconds() / 3600  # Convert to hours
        if action == 1 and total_amount >= increase_amount:  # Increase
          position_amounts.append(increase_amount)
          position_prices.append(price_close_cur)
          positions[-1] += 1
          position_open_timestamps.append(start_time)
          fee_open_amount = increase_amount * fee_open
          realized_pnl -= fee_open_amount
          total_amount -= increase_amount + fee_open_amount
          reward = 0  # Neutral reward, profit will determine later
        elif action == 3 or duration > max_duration_hours or i >= len(df) - step_size:  # Close (or last step, or position duration over threshold)
          fee_close_amount = position_amount_cur * fee_close
          realized_pnl -= fee_close_amount
          total_amount += position_amount_cur - fee_close_amount
          profit = (gross_profit + realized_pnl) / position_amount_sum
          profits.append(profit)
          position_close_timestamps.append(start_time)
          reward = profit * 100  # Reward based on profit percentage
          position_open = False
          position_open_timestamp = None
          position_prices = []
          position_amounts = []
          realized_pnl = 0
        else:  # Hold
          reward = 5 if gross_profit > 0 else -5  # Simplified hold reward

      if train and i < len(df) - step_size:
        next_state = get_state(df, i + step_size)
        done = (i >= len(df) - step_size * 2)
        agent.remember(state, action, reward, next_state, done)
        if len(agent.memory) > batch_size:
          agent.replay(batch_size)

    if train and agent.epsilon > agent.epsilon_min:
      agent.epsilon *= agent.epsilon_decay

    total_profits.append(profits)
    all_positions.append(positions)
    all_open_timestamps.append(position_open_timestamps)
    all_close_timestamps.append(position_close_timestamps)

    if train:
      print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Episode {episode + 1} / {episodes}, Profit: {100 * (total_amount - amount) / amount:.2f}%, Trades: {len(profits)}, Epsilon: {agent.epsilon:.4f}')
    else:
      return total_amount, profits, positions, position_open_timestamps, position_close_timestamps

if __name__ == '__main__':
  symbol = 'BROCCOLIUSDT'
  interval = '5'
  time_start = '2025-02-21'
  amount = 10000
  fee_open = 0.0002
  fee_close = 0.0002
  step_size = 12  # Step every hour
  max_duration_hours = 7 * 24

  # Load data
  df_data_kline = load_data(f'data/kline/{symbol}/{interval}')
  df_data_funding_rate = load_data(f'data/funding_rate/{symbol}')
  df_data_kline['startTime'] = pd.to_datetime(df_data_kline['startTime'], unit='ms')
  df_data_funding_rate['fundingRateTimestamp'] = pd.to_datetime(df_data_funding_rate['fundingRateTimestamp'], unit='ms')
  df_data = pd.merge(df_data_kline, df_data_funding_rate, how='left', left_on='startTime', right_on='fundingRateTimestamp')
  df_data = df_data[df_data['startTime'] > pd.to_datetime(time_start)]
  df_data['closePrice_diff_1h'] = df_data['closePrice'].pct_change(periods=12)
  df_data['rsi_1h'] = calculate_rsi(df_data, period=12)

  # Initialize DQN
  state_size = 5  # RSI, price diff, funding rate, volume, turnover
  agent = DQNAgent(state_size)

  # Train the agent
  episodes = 50
  run_bot(df_data, amount, fee_open, fee_close, agent, train=True, episodes=episodes, start_idx=12, step_size=step_size, max_duration_hours=max_duration_hours)

  # Save the model
  agent.model.save('models/model_broccoli_trading.keras')
  print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Model saved')

  # Test the trained agent
  total_amount, profits, positions, position_open_timestamps, position_close_timestamps = run_bot(df_data, amount, fee_open, fee_close, agent, train=False, step_size=step_size)
  total_profit = (total_amount - amount) / amount
  total_trade_profit = sum(profits) if profits else 0.0
  avg_profit_per_trade = total_trade_profit / len(positions) if positions else 0

  # Calculate position durations
  position_durations = []
  previous_close_time = None
  for close_time in position_close_timestamps:
    open_times_in_range = [open_time for open_time in position_open_timestamps if open_time <= close_time and (previous_close_time is None or open_time >= previous_close_time)]
    if open_times_in_range:
      position_durations.append((close_time - min(open_times_in_range)))
    previous_close_time = close_time

  # Metrics
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
    expected_value = (p_win * avg_profit) - (p_loss * avg_loss)

  print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: \nResult')
  print(f'Total Amount: {total_amount}')
  print(f'Position Total: {len(positions)}')
  print(f'Position Increasing: {positions}')
  print(f'Max Position Increasing: {max(positions) if positions else 0}')
  print(f'Position Durations: {[str(pd.to_timedelta(duration, unit="s")) for duration in position_durations]}')
  print(f'Max Position Duration: {str(pd.to_timedelta(max(position_durations, default=pd.Timedelta(0)).total_seconds(), unit="s"))}')
  print(f'Total profit/loss: {100 * total_profit:.2f} %')
  print(f'Total trade profit/loss: {100 * total_trade_profit:.2f} %')
  print(f'Avg profit per trade: {100 * avg_profit_per_trade:.2f} %')
  print(f'Expected Value (EV): {100 * expected_value:.4f} %')
  print(f'Profit Factor: {profit_factor:.4f}')

  # Plot
  plt.figure(figsize=(12, 6))
  plt.plot(df_data['startTime'], df_data['closePrice'], label='Close Price', color='blue')
  open_prices = [df_data[df_data['startTime'] == ts]['closePrice'].values[0] for ts in position_open_timestamps]
  plt.scatter(position_open_timestamps, open_prices, color='green', label='Position Open', marker='o')
  close_prices = [df_data[df_data['startTime'] == ts]['closePrice'].values[0] for ts in position_close_timestamps]
  plt.scatter(position_close_timestamps, close_prices, color='red', label='Position Close', marker='x')
  plt.xlabel('Timestamp')
  plt.ylabel('Close Price')
  plt.title('Close Price with Position Open/Close Points')
  plt.legend()
  plt.grid()
  plt.show()