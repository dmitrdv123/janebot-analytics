import pandas as pd
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

def set_initial_parameters(volatility_level):
  '''
  Set initial profit target (p) and stop loss (q) based on volatility.
  Values are in price units (e.g., USDT for BTC/USDT).
  '''
  if volatility_level == 0:  # Low volatility
    p = 10  # Profit target
    q = 10  # Stop loss
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

def get_sliding_states(data, start_idx, end_idx, sequence_length, state_size):
  '''Helper for multi-step sequences (optional).'''
  n_samples = end_idx - start_idx
  states = np.zeros((n_samples, sequence_length * state_size))
  for i in range(n_samples):
    t = start_idx + i
    seq_start = max(0, t - sequence_length + 1)
    seq = data[seq_start:t+1].flatten()
    states[i, -len(seq):] = seq
  return states

class DDQNAgent:
  '''
  Double DQN agent with feedforward layers.
  '''
  def __init__(self, state_size, action_size, sequence_length=1, learning_rate=0.001,
               discount_factor=0.9, epsilon=1.0, epsilon_decay=0.95, epsilon_min=0.01,
               batch_size=32, memory_size=2000, target_update_freq=100):
    self.state_size = state_size * sequence_length  # Adjust for sequence_length
    self.action_size = action_size
    self.sequence_length = sequence_length
    self.memory = deque(maxlen=memory_size)
    self.gamma = discount_factor
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min
    self.batch_size = batch_size
    self.target_update_freq = target_update_freq
    self.step_counter = 0
    # Build online and target networks
    self.model = self._build_model(learning_rate)
    self.target_model = self._build_model(learning_rate)
    self.update_target_model()  # Initialize target model with online weights

  def _build_model(self, learning_rate):
    '''
    Build a feedforward neural network for DDQN.
    '''
    model = Sequential()
    model.add(Input(shape=(self.state_size,)))  # Flattened input: sequence_length * state_size
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(self.action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model

  def update_target_model(self):
    '''
    Copy weights from online model to target model.
    '''
    self.target_model.set_weights(self.model.get_weights())

  def remember(self, state, action, reward, next_state, done):
    '''
    Store experience in memory (state and next_state are sequences).
    '''
    self.memory.append((state, action, reward, next_state, done))

  def choose_action(self, state):
    '''
    Choose an action using epsilon-greedy policy with sequence input.
    '''
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    q_values = self.model.predict(state, verbose=0)  # State is (1, sequence_length, state_size)
    return np.argmax(q_values[0])

  def replay(self):
    '''
    Train the model using experience replay with DDQN logic.
    '''
    if len(self.memory) < self.batch_size:
      return

    minibatch = random.sample(self.memory, self.batch_size)
    states = np.array([item[0][0] for item in minibatch])  # (batch_size, sequence_length, state_size)
    actions = np.array([item[1] for item in minibatch])
    rewards = np.array([item[2] for item in minibatch])
    next_states = np.array([item[3][0] for item in minibatch])
    dones = np.array([item[4] for item in minibatch])

    # Predict Q-values with online and target models
    q_values_next = self.model.predict(next_states, verbose=0)  # Online model selects actions
    q_values_target = self.target_model.predict(next_states, verbose=0)  # Target model evaluates

    targets = self.model.predict(states, verbose=0)
    for i in range(self.batch_size):
      if dones[i]:
        targets[i][actions[i]] = rewards[i]
      else:
        # DDQN: Use online model to select action, target model to evaluate
        action_next = np.argmax(q_values_next[i])
        targets[i][actions[i]] = rewards[i] + self.gamma * q_values_target[i][action_next]

    self.model.fit(states, targets, epochs=1, verbose=0)

    # Update epsilon and target model
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
    self.step_counter += 1
    if self.step_counter % self.target_update_freq == 0:
      self.update_target_model()

def simulate_trading(df, agent, episodes=50, fee_open=0.0002, fee_close=0.0002, max_lookahead_seconds=20):
  '''
  Optimized simulation of trading episodes with batched predictions and vectorized operations.
  '''
  feature_cols = ['price', 'returns', 'short_ma', 'long_ma', 'volatility', 'macd_hist', 
          'rsi', 'imbalance', 'velocity', 'trend_direction', 'trend_strength', 'amount_z']

  df = df.dropna(subset=feature_cols).reset_index(drop=True)
  n = len(df)

  end_idx = n - max_lookahead_seconds
  if end_idx <= 0:
    print("Insufficient data for simulation.")
    return

  # Pre-convert to NumPy arrays
  data = df[feature_cols].values  # Shape: (n, 12)
  prices = df['price'].values
  trends = df['trend_direction'].values
  volatilities = df['volatility_level'].values

  for episode in range(episodes):
    print(f'{pd.Timestamp.now()}: Train Episode {episode + 1} / {episodes}')

    # Batch compute states (all time steps)
    states = data[:-max_lookahead_seconds]  # Shape: (end_idx, 12)

    # Batch predict Q-values and apply epsilon-greedy
    q_values = agent.model.predict(states, batch_size=32, verbose=0)
    actions = np.zeros(end_idx, dtype=int)
    rand_mask = np.random.rand(end_idx) <= agent.epsilon
    actions[rand_mask] = np.random.randint(0, agent.action_size, size=rand_mask.sum())
    actions[~rand_mask] = np.argmax(q_values[~rand_mask], axis=1)

    # Vectorized p and q initialization
    p, q = np.vectorize(set_initial_parameters, otypes=[float, float])(volatilities[:-max_lookahead_seconds])
    p = p.copy()
    q = q.copy()

    # Vectorized p and q adjustments
    decrease_p = np.isin(actions, [0, 1, 2])
    increase_p = np.isin(actions, [6, 7, 8])
    decrease_q = np.isin(actions, [0, 3, 6])
    increase_q = np.isin(actions, [2, 5, 8])
    p[decrease_p] = np.maximum(1, p[decrease_p] - 5)
    p[increase_p] += 5
    q[decrease_q] = np.maximum(1, q[decrease_q] - 10)
    q[increase_q] += 10

    # Dynamic lookahead
    lookahead_seconds = np.vectorize(get_lookahead_seconds, otypes=[int])(volatilities[:-max_lookahead_seconds], max_lookahead_seconds)

    # Pre-compute trade outcomes
    X = prices[:-max_lookahead_seconds]
    fees = (fee_open + fee_close) * X
    trend = trends[:-max_lookahead_seconds]
    rewards = np.zeros(end_idx)
    exit_bars = np.zeros(end_idx, dtype=int)

    for t in range(end_idx):
      look_end = min(t + lookahead_seconds[t] + 1, n)
      price_window = prices[t + 1:look_end]
      if trend[t] == 1:  # Long
        hit_loss = price_window <= X[t] - q[t]
        hit_profit = price_window >= X[t] + p[t]
      else:  # Short
        hit_loss = price_window >= X[t] + q[t]
        hit_profit = price_window <= X[t] - p[t]
      loss_idx = np.argmax(hit_loss) if np.any(hit_loss) else -1
      profit_idx = np.argmax(hit_profit) if np.any(hit_profit) else -1
      if loss_idx >= 0 and (profit_idx == -1 or loss_idx < profit_idx):
        rewards[t] = -q[t] - fees[t]
        exit_bars[t] = t + 1 + loss_idx
      elif profit_idx >= 0:
        rewards[t] = p[t] - fees[t]
        exit_bars[t] = t + 1 + profit_idx
      else:
        exit_bar = min(t + lookahead_seconds[t], n - 1)
        rewards[t] = (prices[exit_bar] - X[t]) - fees[t] if trend[t] == 1 else (X[t] - prices[exit_bar]) - fees[t]
        exit_bars[t] = exit_bar

    # Diagnostics
    total_reward = rewards.sum()
    avg_reward = rewards.mean()
    win_trades = (rewards > 0).sum()
    loss_trades = (rewards < 0).sum()
    avg_win = rewards[rewards > 0].mean() if win_trades > 0 else 0
    avg_loss = rewards[rewards < 0].mean() if loss_trades > 0 else 0
    print(f'{pd.Timestamp.now()}: Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, '
        f'Wins: {win_trades}, Losses: {loss_trades}, Avg Win: {avg_win:.2f}, Avg Loss: {avg_loss:.2f}')

    next_states = data[exit_bars]
    # next_states = get_sliding_states(data, min(exit_bars), max(exit_bars) + 1, agent.sequence_length, len(feature_cols))
    # next_states = next_states[exit_bars - min(exit_bars)]
    dones = exit_bars >= (n - 1)
    for t in range(end_idx):
      agent.remember(states[t:t+1], actions[t], rewards[t], next_states[t:t+1], dones[t])

    agent.replay()
    print(f'{pd.Timestamp.now()}: Epsilon: {agent.epsilon:.2f}')

def evaluate_agent(df, agent, fee_open=0.0002, fee_close=0.0002, max_lookahead_seconds=20):
  '''
  Optimized evaluation of the trained agent with batched predictions and vectorized operations.
  '''
  feature_cols = ['price', 'returns', 'short_ma', 'long_ma', 'volatility', 'macd_hist',
          'rsi', 'imbalance', 'velocity', 'trend_direction', 'trend_strength', 'amount_z']
  df = df.dropna(subset=feature_cols).reset_index(drop=True)
  n = len(df)
  end_idx = n - max_lookahead_seconds
  if end_idx <= 0:
    print("Insufficient data for evaluation.")
    return

  # Pre-convert to NumPy arrays
  data = df[feature_cols].values  # Shape: (n, 12)
  prices = df['price'].values
  trends = df['trend_direction'].values
  volatilities = df['volatility'].values

  states = data[:-max_lookahead_seconds]
  # states = get_sliding_states(data, 0, end_idx, agent.sequence_length, len(feature_cols))
  actions = np.argmax(agent.model.predict(states, batch_size=32, verbose=0), axis=1)

  # Vectorized p and q
  p, q = np.vectorize(set_initial_parameters, otypes=[float, float])(volatilities[:-max_lookahead_seconds])
  p = p.copy()
  q = q.copy()

  decrease_p = np.isin(actions, [0, 1, 2])
  increase_p = np.isin(actions, [6, 7, 8])
  decrease_q = np.isin(actions, [0, 3, 6])
  increase_q = np.isin(actions, [2, 5, 8])
  p[decrease_p] = np.maximum(1, p[decrease_p] - 5)
  p[increase_p] += 5
  q[decrease_q] = np.maximum(1, q[decrease_q] - 10)
  q[increase_q] += 10

  # Dynamic lookahead
  lookahead_seconds = np.vectorize(get_lookahead_seconds, otypes=[int])(volatilities[:-max_lookahead_seconds], max_lookahead_seconds)

  # Compute profits
  X = prices[:-max_lookahead_seconds]
  fees = (fee_open + fee_close) * X
  trend = trends[:-max_lookahead_seconds]
  profits = np.zeros(end_idx)

  for t in range(end_idx):
    look_end = min(t + lookahead_seconds[t] + 1, n)
    price_window = prices[t + 1:look_end]
    if trend[t] == 1:  # Long
      hit_loss = price_window <= X[t] - q[t]
      hit_profit = price_window >= X[t] + p[t]
    else:  # Short
      hit_loss = price_window >= X[t] + q[t]
      hit_profit = price_window <= X[t] - p[t]
    loss_idx = np.argmax(hit_loss) if np.any(hit_loss) else -1
    profit_idx = np.argmax(hit_profit) if np.any(hit_profit) else -1
    if loss_idx >= 0 and (profit_idx == -1 or loss_idx < profit_idx):
      profits[t] = -q[t] - fees[t]
    elif profit_idx >= 0:
      profits[t] = p[t] - fees[t]
    else:
      exit_price = prices[min(t + lookahead_seconds[t], n - 1)]
      profits[t] = (exit_price - X[t]) - fees[t] if trend[t] == 1 else (X[t] - exit_price) - fees[t]

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

  # Define RL parameters
  state_size = 12
  action_size = 9
  sequence_length = 1  # Matches window_long from calc_features_trades
  episodes = 10

  fee_open = 0
  fee_close = 0

  # Load features
  df_features_trades = pd.read_csv(f'{folder_features_trades}/features.csv')
  df_features_trades['timestamp'] = pd.to_datetime(df_features_trades['timestamp'], unit='s')
  df_features_trades = df_features_trades.dropna()  # Drop rows with NaN in selected features

  # Create agent
  agent = DDQNAgent(state_size, action_size, sequence_length=sequence_length)

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

      # Train the agent
      simulate_trading(df_features_trades_day, agent, episodes=episodes, fee_open=fee_open, fee_close=fee_close)

      # Evaluate the agent
      evaluate_agent(df_features_trades_day, agent, fee_open=fee_open, fee_close=fee_close)

    current_timestamp = next_timestamp
