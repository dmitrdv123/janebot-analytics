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

class DDQNAgent:
  '''
  Double DQN agent with feedforward layers.
  '''
  def __init__(self, state_size, action_size, sequence_length=60, learning_rate=0.001,
               discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
               batch_size=32, memory_size=2000, target_update_freq=100):
    self.state_size = state_size
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

def get_state_sequence(df, t, sequence_length, feature_cols):
  '''
  Extract a flattened state ending at index t.
  '''
  start_idx = max(0, t - sequence_length + 1)
  sequence = df[feature_cols].iloc[start_idx:t+1].values.flatten()
  if len(sequence) < sequence_length * len(feature_cols):
    # Pad with zeros if sequence is shorter than sequence_length
    padding = np.zeros(sequence_length * len(feature_cols) - len(sequence))
    sequence = np.concatenate((padding, sequence))
  return sequence.reshape(1, -1)  # Shape: (1, sequence_length * state_size)

def simulate_trading(df, agent, episodes=50, fee_open=0.0002, fee_close=0.0002, max_lookahead_seconds=20):
  '''
  Simulate trading episodes to train the RDDQN agent with sequences.
  '''
  feature_cols = ['price', 'returns', 'short_ma', 'long_ma', 'volatility', 'macd_hist', 
                  'rsi', 'imbalance', 'velocity', 'trend_direction', 'trend_strength', 'amount_z']
  for episode in range(episodes):
    print(f'{pd.Timestamp.now()}: Train Episode {episode + 1} / {episodes}')

    rewards = np.array([])

    for t in range(agent.sequence_length, len(df) - max_lookahead_seconds):
      state = get_state_sequence(df, t, agent.sequence_length, feature_cols)
      action = agent.choose_action(state)

      volatility_level = df['volatility_level'].iloc[t]
      p, q = set_initial_parameters(volatility_level)
      lookahead_seconds = get_lookahead_seconds(volatility_level, max_lookahead_seconds=max_lookahead_seconds)

      # Adjust p and q based on action
      if action in [0, 1, 2]: p = max(1, p - 5)
      elif action in [6, 7, 8]: p += 5
      if action in [0, 3, 6]: q = max(1, q - 10)
      elif action in [2, 5, 8]: q += 10

      X = df['price'].iloc[t]
      trend = df['trend_direction'].iloc[t]
      fees = (fee_open + fee_close) * X  # open + close fee

      if trend == 1:  # Long position
        for k in range(t + 1, min(t + lookahead_seconds + 1, len(df))):
          if df['price'].iloc[k] <= X - q:
            reward = -q - fees
            exit_bar = k
            break
          elif df['price'].iloc[k] >= X + p:
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
          elif df['price'].iloc[k] <= X - p:
            reward = p - fees
            exit_bar = k
            break
        else:
          exit_bar = min(t + lookahead_seconds, len(df) - 1)
          reward = (X - df['price'].iloc[exit_bar]) - fees

      next_state = get_state_sequence(df, exit_bar, agent.sequence_length, feature_cols)
      done = exit_bar >= len(df) - 1
      agent.remember(state, action, reward, next_state, done)
      rewards = np.append(rewards, reward)

    agent.replay()

    total_reward = rewards.sum()
    avg_reward = rewards.mean()
    win_trades = (rewards > 0).sum()
    loss_trades = (rewards < 0).sum()
    avg_win = rewards[rewards > 0].mean() if win_trades > 0 else 0
    avg_loss = rewards[rewards < 0].mean() if loss_trades > 0 else 0
    print(f'{pd.Timestamp.now()}: Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, '
        f'Wins: {win_trades}, Losses: {loss_trades}, Avg Win: {avg_win:.2f}, Avg Loss: {avg_loss:.2f}, Epsilon: {agent.epsilon:.2f}')

def evaluate_agent(df, agent, fee_open=0.0002, fee_close=0.0002, max_lookahead_seconds=20):
  '''
  Evaluate the trained agent's performance with flattened states.
  '''
  feature_cols = ['price', 'returns', 'short_ma', 'long_ma', 'volatility', 'macd_hist',
                  'rsi', 'imbalance', 'velocity', 'trend_direction', 'trend_strength', 'amount_z']
  df = df.dropna(subset=feature_cols)
  
  profits = np.array([])

  for t in range(agent.sequence_length, len(df) - max_lookahead_seconds):  # Max lookahead = 20s
    state = get_state_sequence(df, t, agent.sequence_length, feature_cols)
    action = np.argmax(agent.model.predict(state, verbose=0)[0])

    volatility = df['volatility'].iloc[t]
    p, q = set_initial_parameters(volatility)
    lookahead_seconds = get_lookahead_seconds(volatility, max_lookahead_seconds=max_lookahead_seconds)

    if action in [0, 1, 2]: p = max(1, p - 5)
    elif action in [6, 7, 8]: p += 5
    if action in [0, 3, 6]: q = max(1, q - 10)
    elif action in [2, 5, 8]: q += 10

    X = df['price'].iloc[t]
    trend = df['trend_direction'].iloc[t]
    fees = (fee_open + fee_close) * X  # open + close fee

    if trend == 1:  # Long position
      for k in range(t + 1, min(t + lookahead_seconds + 1, len(df))):
        if df['price'].iloc[k] <= X - q:
          profit = -q - fees
          break
        elif df['price'].iloc[k] >= X + p:
          profit = p - fees
          break
      else:
        profit = (df['price'].iloc[min(t + lookahead_seconds, len(df) - 1)] - X) - fees
    else:  # Short position
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
