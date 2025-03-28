import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LayerNormalization, MultiHeadAttention, Dropout, Layer
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

def set_initial_parameters(volatility_level):
    if volatility_level == 0:
        p = 10
        q = 10
    elif volatility_level == 1:
        p = 30
        q = 30
    else:
        p = 50
        q = 50
    return p, q

def get_lookahead_seconds(volatility_level, max_lookahead_seconds=20):
    if volatility_level == 0: return 10
    elif volatility_level == 1: return 15
    else: return max_lookahead_seconds

def get_state_sequence(df, t, sequence_length, feature_cols):
    start_idx = max(0, t - sequence_length + 1)
    sequence = df[feature_cols].iloc[start_idx:t+1].values
    if sequence.shape[0] < sequence_length:
        padding = np.zeros((sequence_length - sequence.shape[0], len(feature_cols)))
        sequence = np.vstack((padding, sequence))
    return sequence[np.newaxis, ...]  # Shape: (1, sequence_length, state_size)

class MeanPoolingLayer(Layer):
    def __init__(self, **kwargs):
        super(MeanPoolingLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)  # Pool over sequence dimension

class TransformerPPOAgent:
    def __init__(self, state_size, action_size, sequence_length=1, learning_rate=0.0003,
                 gamma=0.9, clip_ratio=0.2, value_loss_coef=0.5, entropy_coef=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.actor, self.critic = self._build_models()
        self.actor_optimizer = Adam(learning_rate)
        self.critic_optimizer = Adam(learning_rate)
        self.memory = []

    def _build_transformer_block(self, inputs, num_heads=4, dff=128, dropout_rate=0.1):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=self.state_size)(inputs, inputs)
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
        ffn_output = Dense(dff, activation='relu')(out1)
        ffn_output = Dense(self.state_size)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    def _build_models(self):
        # Actor Network
        state_input = Input(shape=(self.sequence_length, self.state_size))
        x = self._build_transformer_block(state_input)
        x = MeanPoolingLayer()(x)  # Custom layer for pooling
        x = Dense(64, activation='relu')(x)
        logits = Dense(self.action_size)(x)
        actor = Model(inputs=state_input, outputs=logits)

        # Critic Network
        critic_input = Input(shape=(self.sequence_length, self.state_size))
        y = self._build_transformer_block(critic_input)
        y = MeanPoolingLayer()(y)
        y = Dense(64, activation='relu')(y)
        value = Dense(1)(y)
        critic = Model(inputs=critic_input, outputs=value)

        return actor, critic

    def choose_action(self, state):
        logits = self.actor.predict(state, verbose=0)[0]
        dist = tfp.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action), log_prob

    def remember(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        return advantages, returns

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones, old_log_probs = zip(*self.memory)
        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.vstack(next_states)
        dones = np.array(dones)
        old_log_probs = np.array(old_log_probs)

        values = self.critic.predict(states, verbose=0).flatten()
        next_values = self.critic.predict(next_states, verbose=0).flatten()
        advantages, returns = self.compute_advantages(rewards, values, next_values, dones)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            logits = self.actor(states)
            dist = tfp.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            ratios = tf.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - self.entropy_coef * tf.reduce_mean(entropy)

            predicted_values = self.critic(states)[:, 0]
            value_loss = tf.reduce_mean(tf.square(returns - predicted_values))

            total_loss = actor_loss + self.value_loss_coef * value_loss

        actor_grads = actor_tape.gradient(total_loss, self.actor.trainable_variables)
        critic_grads = critic_tape.gradient(value_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        self.memory.clear()

def simulate_trading(df, agent, episodes=10, fee_open=0.0002, fee_close=0.0002, max_lookahead_seconds=20):
    feature_cols = ['price', 'returns', 'short_ma', 'long_ma', 'volatility', 'macd_hist', 
                    'rsi', 'imbalance', 'velocity', 'trend_direction', 'trend_strength', 'amount_z']
    for episode in range(episodes):
        print(f'{pd.Timestamp.now()}: Train Episode {episode + 1} / {episodes}')
        rewards = np.array([])

        for t in range(agent.sequence_length, len(df) - max_lookahead_seconds):
            if t % 100 == 0:
              print(f'{pd.Timestamp.now()}: Iteration {t} / {len(df) - max_lookahead_seconds}')
          
            state = get_state_sequence(df, t, agent.sequence_length, feature_cols)
            action, log_prob = agent.choose_action(state)

            volatility_level = df['volatility_level'].iloc[t]
            p, q = set_initial_parameters(volatility_level)
            lookahead_seconds = get_lookahead_seconds(volatility_level, max_lookahead_seconds)

            if action in [0, 1, 2]: p = max(1, p - 5)
            elif action in [6, 7, 8]: p += 5
            if action in [0, 3, 6]: q = max(1, q - 10)
            elif action in [2, 5, 8]: q += 10

            X = df['price'].iloc[t]
            trend = df['trend_direction'].iloc[t]
            fees = (fee_open + fee_close) * X

            if trend == 1:
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
            else:
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
            agent.remember(state, action, reward, next_state, done, log_prob)
            rewards = np.append(rewards, reward)

        print(f'{pd.Timestamp.now()}: Train agent')
        agent.train()

        total_reward = rewards.sum()
        avg_reward = rewards.mean()
        win_trades = (rewards > 0).sum()
        loss_trades = (rewards < 0).sum()
        avg_win = rewards[rewards > 0].mean() if win_trades > 0 else 0
        avg_loss = rewards[rewards < 0].mean() if loss_trades > 0 else 0
        print(f'{pd.Timestamp.now()}: Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, '
              f'Wins: {win_trades}, Losses: {loss_trades}, Avg Win: {avg_win:.2f}, Avg Loss: {avg_loss:.2f}')

def evaluate_agent(df, agent, fee_open=0.0002, fee_close=0.0002, max_lookahead_seconds=20):
    feature_cols = ['price', 'returns', 'short_ma', 'long_ma', 'volatility', 'macd_hist',
                    'rsi', 'imbalance', 'velocity', 'trend_direction', 'trend_strength', 'amount_z']
    df = df.dropna(subset=feature_cols)
    profits = np.array([])

    for t in range(agent.sequence_length, len(df) - max_lookahead_seconds):
        state = get_state_sequence(df, t, agent.sequence_length, feature_cols)
        action, _ = agent.choose_action(state)

        volatility_level = df['volatility_level'].iloc[t]
        p, q = set_initial_parameters(volatility_level)
        lookahead_seconds = get_lookahead_seconds(volatility_level, max_lookahead_seconds)

        if action in [0, 1, 2]: p = max(1, p - 5)
        elif action in [6, 7, 8]: p += 5
        if action in [0, 3, 6]: q = max(1, q - 10)
        elif action in [2, 5, 8]: q += 10

        X = df['price'].iloc[t]
        trend = df['trend_direction'].iloc[t]
        fees = (fee_open + fee_close) * X

        if trend == 1:
            for k in range(t + 1, min(t + lookahead_seconds + 1, len(df))):
                if df['price'].iloc[k] <= X - q:
                    profit = -q - fees
                    break
                elif df['price'].iloc[k] >= X + p:
                    profit = p - fees
                    break
            else:
                profit = (df['price'].iloc[min(t + lookahead_seconds, len(df) - 1)] - X) - fees
        else:
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
    state_size = 12
    action_size = 9
    sequence_length = 1
    episodes = 10
    fee_open = 0
    fee_close = 0

    df_features_trades = pd.read_csv(f'{folder_features_trades}/features.csv')
    df_features_trades['timestamp'] = pd.to_datetime(df_features_trades['timestamp'], unit='s')
    df_features_trades = df_features_trades.dropna()

    agent = TransformerPPOAgent(state_size, action_size, sequence_length=sequence_length)

    min_timestamp = df_features_trades['timestamp'].min()
    max_timestamp = df_features_trades['timestamp'].max()

    current_timestamp = min_timestamp
    while current_timestamp <= max_timestamp:
        next_timestamp = current_timestamp + pd.Timedelta(days=1)
        df_features_trades_day = df_features_trades[
            (df_features_trades['timestamp'] >= current_timestamp) & 
            (df_features_trades['timestamp'] < next_timestamp)
        ]

        if not df_features_trades_day.empty:
            print(f'{pd.Timestamp.now()}: Processing data from {current_timestamp.date()}')
            simulate_trading(df_features_trades_day, agent, episodes=episodes, fee_open=fee_open, fee_close=fee_close)
            evaluate_agent(df_features_trades_day, agent, fee_open=fee_open, fee_close=fee_close)

        current_timestamp = next_timestamp