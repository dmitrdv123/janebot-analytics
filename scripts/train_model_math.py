import numpy as np
import pandas as pd
import random

def set_initial_parameters(volatility_level):
  '''
  Set initial profit target (p) and stop loss (q) based on volatility.
  Values are in price units (e.g., USDT for BTC/USDT).
  '''
  if volatility_level == 0:  # Low volatility
    p = 10
    q = 10
  elif volatility_level == 1:  # Moderate volatility
    p = 30
    q = 30
  else:  # High volatility
    p = 50
    q = 50
  return p, q

def get_lookahead_seconds(volatility_level, max_lookahead_seconds=20):
  '''Dynamic lookahead based on volatility.'''
  if volatility_level == 0: return 10
  elif volatility_level == 1: return 15
  else: return max_lookahead_seconds

def choose_action():
  '''
  Mathematical model: Randomly choose an action (0-8) to adjust p and q.
  Mimics high epsilon behavior from your observation.
  '''
  return random.randrange(9)  # 0 to 8, matching original action_size

def simulate_trading(df, episodes=50, fee_open=0.0002, fee_close=0.0002, max_lookahead_seconds=20):
  '''
  Simulate trading episodes with a random mathematical model.
  '''
  feature_cols = ['price', 'returns', 'short_ma', 'long_ma', 'volatility', 'macd_hist', 
          'rsi', 'imbalance', 'velocity', 'trend_direction', 'trend_strength', 'amount_z']
  df = df.dropna(subset=feature_cols).reset_index(drop=True)
  
  for episode in range(episodes):
    print(f'{pd.Timestamp.now()}: Train Episode {episode + 1} / {episodes}')

    rewards = np.array([])

    for t in range(0, len(df) - max_lookahead_seconds):
      action = choose_action()  # Random action

      volatility_level = df['volatility_level'].iloc[t]
      p, q = set_initial_parameters(volatility_level)
      lookahead_seconds = get_lookahead_seconds(volatility_level, max_lookahead_seconds=max_lookahead_seconds)
      
      # Adjust p and q based on action (same logic as before)
      if action in [0, 1, 2]: p = max(1, p - 5)
      elif action in [6, 7, 8]: p += 5
      if action in [0, 3, 6]: q = max(1, q - 10)
      elif action in [2, 5, 8]: q += 10

      X = df['price'].iloc[t]
      trend = df['trend_direction'].iloc[t]
      fees = (fee_open + fee_close) * X

      if trend == 1:  # Long position
        for k in range(t + 1, min(t + lookahead_seconds + 1, len(df))):
          if df['price'].iloc[k] <= X - q:
            reward = -q - fees
            exit_bar = k
            break
          elif df['price'].iloc[k] >= X + p + fees:
            reward = p - fees
            exit_bar = k
            break
        else:
          exit_bar = min(t + lookahead_seconds, len(df) - 1)
          reward = (df['price'].iloc[exit_bar] - X) - fees
      else:  # Short position
        for k in range(t + 1, min(t + lookahead_seconds + 1, len(df))):
          if df['price'].iloc[k] >= X + q:
            reward = -q - fees
            exit_bar = k
            break
          elif df['price'].iloc[k] <= X - p - fees:
            reward = p - fees
            exit_bar = k
            break
        else:
          exit_bar = min(t + lookahead_seconds, len(df) - 1)
          reward = (X - df['price'].iloc[exit_bar]) - fees

      rewards = np.append(rewards, reward)

    total_reward = rewards.sum()
    avg_reward = rewards.mean()
    win_trades = (rewards > 0).sum()
    loss_trades = (rewards < 0).sum()
    avg_win = rewards[rewards > 0].mean() if win_trades > 0 else 0
    avg_loss = rewards[rewards < 0].mean() if loss_trades > 0 else 0
    print(f'{pd.Timestamp.now()}: Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, '
        f'Wins: {win_trades}, Losses: {loss_trades}, Avg Win: {avg_win:.2f}, Avg Loss: {avg_loss:.2f}')

def evaluate_agent(df, fee_open=0.0002, fee_close=0.0002, max_lookahead_seconds=20):
  '''
  Evaluate the random mathematical model's performance.
  '''
  feature_cols = ['price', 'returns', 'short_ma', 'long_ma', 'volatility', 'macd_hist',
          'rsi', 'imbalance', 'velocity', 'trend_direction', 'trend_strength', 'amount_z']
  df = df.dropna(subset=feature_cols).reset_index(drop=True)
  total_profit = 0
  wins = 0
  losses = 0
  profits = np.array([])

  for t in range(0, len(df) - max_lookahead_seconds):
    action = choose_action()  # Random action

    volatility_level = df['volatility_level'].iloc[t]
    p, q = set_initial_parameters(volatility_level)
    lookahead_seconds = get_lookahead_seconds(volatility_level, max_lookahead_seconds=max_lookahead_seconds)

    if action in [0, 1, 2]: p = max(1, p - 5)
    elif action in [6, 7, 8]: p += 5
    if action in [0, 3, 6]: q = max(1, q - 10)
    elif action in [2, 5, 8]: q += 10

    X = df['price'].iloc[t]
    trend = df['trend_direction'].iloc[t]
    fees = (fee_open + fee_close) * X

    if trend == 1:  # Long
      for k in range(t + 1, min(t + lookahead_seconds + 1, len(df))):
        if df['price'].iloc[k] <= X - q:
          profit = -q - fees
          break
        elif df['price'].iloc[k] >= X + p:
          profit = p - fees
          break
      else:
        profit = (df['price'].iloc[min(t + lookahead_seconds, len(df) - 1)] - X) - fees
    else:  # Short
      for k in range(t + 1, min(t + lookahead_seconds + 1, len(df))):
        if df['price'].iloc[k] >= X + q:
          profit = -q - fees
          break
        elif df['price'].iloc[k] <= X - p:
          profit = p - fees
          break
      else:
        profit = (X - df['price'].iloc[min(t + lookahead_seconds, len(df) - 1)]) - fees

    profits = np.append(profits, profit)

  total_profit = profits.sum()
  wins = (profits > 0).sum()
  losses = (profits < 0).sum()
  win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
  avg_profit = profits.mean()
  avg_win = profits[profits > 0].mean() if wins > 0 else 0
  avg_loss = profits[profits < 0].mean() if losses > 0 else 0
  print(f'Total Profit: {total_profit:.2f}, Win Rate: {win_rate:.2f}, '
      f'Avg Profit: {avg_profit:.2f}, Avg Win: {avg_win:.2f}, Avg Loss: {avg_loss:.2f}')

if __name__ == '__main__':
  symbol = 'BTCUSDT'
  folder_features_trades = f'features/trades/{symbol}'
  state_size = 12  # Not used directly but kept for context
  action_size = 9  # Number of possible actions
  episodes = 5
  fee_open = 0
  fee_close = 0
  max_lookahead_seconds = 20

  # Load features
  df_features_trades = pd.read_csv(f'{folder_features_trades}/features.csv')
  df_features_trades['timestamp'] = pd.to_datetime(df_features_trades['timestamp'], unit='s')
  df_features_trades = df_features_trades.dropna()

  # Get min and max timestamp
  min_timestamp = df_features_trades['timestamp'].min()
  max_timestamp = df_features_trades['timestamp'].max()

  # Iterate by day interval
  current_timestamp = min_timestamp
  while current_timestamp <= max_timestamp:
    next_timestamp = current_timestamp + pd.Timedelta(days=1)
    df_features_trades_day = df_features_trades[
      (df_features_trades['timestamp'] >= current_timestamp) & 
      (df_features_trades['timestamp'] < next_timestamp)
    ]

    if not df_features_trades_day.empty:
      print(f'{pd.Timestamp.now()}: Processing data from {current_timestamp.date()}')
      simulate_trading(df_features_trades_day, episodes=episodes, fee_open=fee_open, fee_close=fee_close, max_lookahead_seconds=max_lookahead_seconds)
      evaluate_agent(df_features_trades_day, fee_open=fee_open, fee_close=fee_close, max_lookahead_seconds=max_lookahead_seconds)

    current_timestamp = next_timestamp