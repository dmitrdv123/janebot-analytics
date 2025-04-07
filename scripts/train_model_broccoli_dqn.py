from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import gym
from gym import spaces
from collections import deque
import random
import matplotlib.pyplot as plt  # Added for plotting

from utils_features import calculate_rsi
from utils import load_data

# Trading Environment
class TradingEnv(gym.Env):
    def __init__(self, df, initial_amount, fee_open=0.0002, fee_close=0.0002, k_position_amount_open=0.1, k_position_amount_increase=0.05):
        super(TradingEnv, self).__init__()

        self.df = df.to_numpy()  # Convert DataFrame to NumPy array

        column_names = list(df.columns)
        self.rsi_1h_idx = column_names.index('rsi_1h')
        self.close_price_diff_1h_idx = column_names.index('close_price_diff_1h')
        self.close_price_idx = column_names.index('closePrice')
        self.funding_rate_idx = column_names.index('fundingRate')
        self.start_time_idx = column_names.index('startTime')

        self.fee_open = fee_open
        self.fee_close = fee_close
        self.k_position_amount_open = k_position_amount_open
        self.k_position_amount_increase = k_position_amount_increase

        self.initial_amount = initial_amount
        self.total_amount = initial_amount

        self.positions = []  # [{'amount': float, 'price': float, 'open_time': timestamp}]
        self.position_history = []  # [{'open_time': timestamp, 'close_time': timestamp, 'net_profit': float, 'count': int}]
        self.current_step = 0
        self.realized_pnl = 0

        # Track metrics
        self.trade_profits = []  # Store profit/loss per trade
        self.position_durations = []  # Store duration of each position in seconds

        self.action_space = spaces.Discrete(4)  # 0: open, 1: increase, 2: close, 3: hold
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

    def reset(self):
        self.total_amount = self.initial_amount
        self.positions = []
        self.position_history = []
        self.current_step = 0
        self.realized_pnl = 0
        self.trade_profits = []
        self.position_durations = []
        return self._get_observation()

    def step(self, action):
        # Get observation before processing the action
        obs = self._get_observation()
        
        # Reward component weights
        w1 = 100.0  # Weight for total profit
        w2 = 0.5  # Weight for winning trades ratio
        w3 = 0.5  # Weight for penalty on losing trades
        w4 = 1.0  # Weight for number of positions
        w5 = 0.5  # Weight for excessive positions

        # Get current row for action processing
        current_row = self.df[self.current_step]
        current_price = current_row[self.close_price_idx]
        current_time = current_row[self.start_time_idx]
        funding_rate = current_row[self.funding_rate_idx]

        reward = 0
        done = self.current_step >= len(self.df) - 1

        gross_profit = 0
        position_amount_sum = 0
        if self.positions:
            amounts = np.array([pos['amount'] for pos in self.positions])
            prices = np.array([pos['price'] for pos in self.positions])
            gross_profit = np.sum(amounts * (prices - current_price) / prices)
            position_amount_sum = np.sum(amounts)
        total_position_amount = position_amount_sum + gross_profit

        # Account for funding rate
        if total_position_amount > 0 and not np.isnan(funding_rate):
            funding_reward = total_position_amount * funding_rate
            self.total_amount += funding_reward
            self.realized_pnl += funding_reward
            reward += w1 * funding_reward / self.initial_amount

        if action == 0:  # Open position
            reward += self._open_position(current_price, current_time, w1)
        elif action == 1:  # Increase position
            reward += self._increase_position(current_price, current_time, w1)
        elif action == 2 or done:  # Close position
            reward += self._close_position(total_position_amount, gross_profit, current_time, w1)

        # Reward calculation
        if done:
            total_profit = (self.total_amount - self.initial_amount) / self.initial_amount
            n_positions = len(self.position_history) if self.position_history else 1
            n_winning_trades = len([p for p in self.position_history if p['net_profit'] > 0])
            n_losing_trades = len([p for p in self.position_history if p['net_profit'] < 0])

            max_positions = len(self.df) / 10
            min_positions = max_positions / 10

            # Reward formula
            final_reward = 0
            if n_positions > max_positions:
                final_reward -= w5 * (n_positions - max_positions) ** 2 / max_positions
            elif n_positions < min_positions:
                final_reward -= w5 * (n_positions - min_positions) ** 2 / min_positions
            elif n_positions > 0:
                final_reward = (
                    w1 * total_profit +
                    w2 * n_winning_trades / (n_winning_trades + n_losing_trades + 1e-6) -
                    w3 * n_losing_trades / n_positions +
                    w4 * total_profit * n_positions / max_positions
                )
            else:
                final_reward -= w1 * 0.1

            reward += final_reward

        self.current_step += 1

        return obs, reward, done, {}

    def _open_position(self, current_price, current_time, w1):
        if self.positions:
            return 0

        position_amount = min(self.k_position_amount_open * self.initial_amount, self.total_amount)
        if position_amount <= 0:
            return 0

        fee = position_amount * self.fee_open
        self.realized_pnl -= fee
        self.total_amount -= position_amount + fee

        self.positions.append({
            'amount': position_amount,
            'price': current_price,
            'open_time': current_time,
        })
        return -w1 * fee / position_amount

    def _increase_position(self, current_price, current_time, w1):
        if not self.positions:
            return 0

        increase_amount = min(self.k_position_amount_increase * self.initial_amount, self.total_amount)
        if increase_amount <= 0:
            return 0

        fee = increase_amount * self.fee_open
        self.realized_pnl -= fee
        self.total_amount -= increase_amount + fee

        self.positions.append({
            'amount': increase_amount,
            'price': current_price,
            'open_time': current_time,
        })
        return -w1 * fee / increase_amount

    def _close_position(self, total_position_amount, gross_profit, current_time, w1):
        if not self.positions:
            return 0

        fee = total_position_amount * self.fee_close
        net_profit = gross_profit + self.realized_pnl - fee
        self.total_amount += total_position_amount - fee

        # Record trade profit and duration
        self.trade_profits.append(net_profit / total_position_amount)  # Normalize by initial amount
        for pos in self.positions:
            duration = (current_time - pos['open_time']).total_seconds()
            self.position_durations.append(duration)

        self.position_history.append({
            'open_time': self.positions[0]['open_time'],
            'close_time': current_time,
            'net_profit': net_profit,
            'count': len(self.positions)
        })

        self.realized_pnl = 0
        self.positions = []

        return w1 * (gross_profit - fee) / total_position_amount

    def _get_profit(self):
        if not self.positions:
            return 0

        current_price = self.df[self.current_step][self.close_price_idx]

        amounts = np.array([pos['amount'] for pos in self.positions])
        prices = np.array([pos['price'] for pos in self.positions])
        gross_profit = np.sum(amounts * (prices - current_price) / prices)
        position_amount_sum = np.sum(amounts)
        return (gross_profit + self.realized_pnl) / position_amount_sum

    def _get_observation(self):
        row = self.df[self.current_step]
        rsi = row[self.rsi_1h_idx]
        price_diff_1h = row[self.close_price_diff_1h_idx]
        current_price = row[self.close_price_idx]
        funding_rate = row[self.funding_rate_idx] if not np.isnan(row[self.funding_rate_idx]) else 0
        position_profit = self._get_profit()
        duration = (row[self.start_time_idx] - self.positions[0]['open_time']).total_seconds() / 3600 if self.positions else 0
        price_change_since_last_increase = (current_price - self.positions[-1]['price']) / self.positions[-1]['price'] if self.positions else 0
        return np.array([rsi, price_diff_1h, position_profit, duration, self.total_amount, current_price, funding_rate, price_change_since_last_increase], dtype=np.float32)

# State Normalization
def normalize_state(state):
    return (state - np.mean(state)) / (np.std(state) + 1e-8)

# DQN Model
class DQNModel(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQNModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQNModel(action_size)
        self.target_model = DQNModel(action_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = normalize_state(state)
            next_state = normalize_state(next_state)
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])
            target = reward if done else reward + self.gamma * np.amax(self.target_model(next_state)[0])
            target_f = self.model(state).numpy()
            target_f[0][action] = target
            self.train_step(state, target_f)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    @tf.function
    def train_step(self, state, target_f):
        with tf.GradientTape() as tape:
            q_values = self.model(state)
            loss = tf.keras.losses.MSE(target_f, q_values)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

# Training Function
def train_dqn(env, agent, episodes, batch_size, replay_interval=100):
    for e in range(episodes):
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Episode: {e + 1}/{episodes}')

        state = env.reset()
        state = normalize_state(state)
        state = np.reshape(state, [1, agent.state_size])
        total_reward = 0
        for time in range(len(env.df)):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = normalize_state(next_state)
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
            if len(agent.memory) > batch_size and time % replay_interval == 0:
                agent.replay(batch_size)

        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}')

        agent.update_target_model()

# Example Usage
if __name__ == "__main__":
    symbol = 'BROCCOLIUSDT'
    interval = '5'  # Kline interval (1m, 5m, 15m, etc.)
    time_start = '2024-01-01'
    initial_amount = 10000

    fee_open = 0.0002
    fee_close = 0.0002

    # Load data
    df_data_kline = load_data(f'data/kline/{symbol}/{interval}')
    df_data_funding_rate = load_data(f'data/funding_rate/{symbol}')

    df_data_kline['startTime'] = pd.to_datetime(df_data_kline['startTime'], unit='ms')
    df_data_funding_rate['fundingRateTimestamp'] = pd.to_datetime(df_data_funding_rate['fundingRateTimestamp'], unit='ms')

    # Merge data
    df_data = pd.merge(df_data_kline, df_data_funding_rate, how='left', left_on='startTime', right_on='fundingRateTimestamp')
    df_data = df_data[df_data['startTime'] > pd.to_datetime(time_start)]

    # Calculate features
    df_data['close_price_diff_1h'] = df_data['closePrice'].pct_change(periods=12)  # 1 hour = 12 periods
    df_data['rsi_1h'] = calculate_rsi(df_data, period=12)  # 1 hour = 12 periods

    # Drop records without features
    df_data = df_data.iloc[12:].reset_index(drop=True)

    env = TradingEnv(df_data, initial_amount=initial_amount, fee_open=fee_open, fee_close=fee_close)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    episodes = 10
    batch_size = 32

    train_dqn(env, agent, episodes, batch_size)

    # Test the agent
    state = env.reset()
    total_reward = 0
    done = False
    positions = []  # Track number of positions per trade
    profits = []    # Track profits per step (not per trade)

    while not done:
        state = normalize_state(state)
        state = np.reshape(state, [1, state_size])
        action = np.argmax(agent.model(state)[0])
        next_state, reward, done, _ = env.step(action)
        positions.append(len(env.positions))  # Record position count
        profit = env._get_profit() if env.positions else 0  # Current unrealized/realized profit
        profits.append(profit)
        state = next_state
        total_reward += reward

    # Calculate metrics
    total_profit = (env.total_amount - env.initial_amount) / env.initial_amount
    profit_baseline = (df_data['closePrice'].iloc[-1] - df_data['closePrice'].iloc[0]) / df_data['closePrice'].iloc[0]  # Buy-and-hold profit
    total_trade_profit = sum(env.trade_profits) if env.trade_profits else 0
    avg_profit_per_trade = total_trade_profit / len(env.trade_profits) if env.trade_profits else 0
    profit_factor = sum([p for p in env.trade_profits if p > 0]) / abs(sum([p for p in env.trade_profits if p < 0])) if any(p < 0 for p in env.trade_profits) else float('inf')
    expected_value = avg_profit_per_trade * (len([p for p in env.trade_profits if p > 0]) / len(env.trade_profits)) if env.trade_profits else 0
    reward_to_risk_ratio = sum([p for p in env.trade_profits if p > 0]) / abs(sum([p for p in env.trade_profits if p < 0])) if any(p < 0 for p in env.trade_profits) else float('inf')

    # Print statements
    print(f'Positions Count: {len(positions)}')
    print(f'Positions Increasing: {positions}')
    print(f'Max Position Increasing: {max(positions) if positions else 0}')
    print(f'Profits: {profits}')
    print(f'Position Durations: {[str(pd.to_timedelta(duration, unit="s")) for duration in env.position_durations]}')
    print(f'Max Position Duration: {str(pd.to_timedelta(max(env.position_durations, default=pd.Timedelta(0)).total_seconds(), unit="s"))}')
    print(f'Baseline Total Profit/Loss: {100 * profit_baseline:.2f} %')
    print(f'Total Profit/Loss: {100 * total_profit:.2f} %')
    print(f'Total Trade Profit/Loss: {100 * total_trade_profit:.2f} %')
    print(f'Avg Profit per Trade: {100 * avg_profit_per_trade:.2f} %')
    print(f'Profit Factor: {profit_factor:.4f}')
    print(f'Expected Value (EV): {100 * expected_value:.4f} %')
    print(f'Reward to Risk Ratio (RRR): {100 * reward_to_risk_ratio:.4f} %')
    print(f"Total Reward from Testing: {total_reward}")

    # Plotting
    position_open_timestamps = [pos['open_time'] for pos in env.position_history]
    position_close_timestamps = [pos['close_time'] for pos in env.position_history]

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
