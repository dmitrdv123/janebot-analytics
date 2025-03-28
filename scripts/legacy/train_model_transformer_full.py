import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from utils_model import merge_features, run_bot, prepare_features_kline, prepare_features_orderbook, prepare_features_long_short_ratio, prepare_features_funding_rate
from constants import columns_input_kline, columns_input_orderbook, columns_input_long_short_ratio, columns_input_funding_rate

# Function to create sequences for time series data
def create_sequences(data, target, seq_length):
  X, y = [], []
  for i in range(len(data) - seq_length):
    X.append(data[i:i + seq_length])
    y.append(target[i + seq_length])
  return np.array(X), np.array(y)

# Custom Transformer Encoder Layer
class TransformerEncoderLayer(tf.keras.layers.Layer):
  def __init__(self, head_size, num_heads, ff_dim, dropout=0.1):
    super(TransformerEncoderLayer, self).__init__()
    self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)
    self.ffn = tf.keras.Sequential([
      tf.keras.layers.Dense(ff_dim, activation='relu'),
      tf.keras.layers.Dense(head_size * num_heads)  # Match the output dimension for attention
    ])
    self.ffn_projection = None  # Projection layer to match d_model, initialized in build
    self.dropout1 = tf.keras.layers.Dropout(dropout)
    self.dropout2 = tf.keras.layers.Dropout(dropout)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.pos_encoding = None  # Will be initialized in build

  def build(self, input_shape):
    # Initialize positional encoding and projection layer based on input sequence length
    seq_length = input_shape[1]
    d_model = input_shape[-1]  # Use input feature dimension directly
    self.pos_encoding = tf.keras.layers.Embedding(seq_length, d_model)
    self.ffn_projection = tf.keras.layers.Dense(d_model)  # Projection layer to match d_model

  def call(self, inputs, training=False):
    # Get sequence length and model dimension from inputs
    seq_length = tf.shape(inputs)[1]
    d_model = inputs.shape[-1]

    # Positional encoding (using the initialized embedding layer)
    pos_encoding = self.pos_encoding(tf.range(seq_length))
    x = inputs + tf.cast(pos_encoding, dtype=inputs.dtype)  # Ensure type matches inputs

    # Multi-head attention
    attn_output = self.attention(x, x, training=training)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)

    # Feed-forward network
    ffn_output = self.ffn(out1)
    ffn_output = self.ffn_projection(ffn_output)  # Project back to d_model
    ffn_output = self.dropout2(ffn_output, training=training)
    return self.layernorm2(out1 + ffn_output)

  def compute_output_shape(self, input_shape):
    # Define the output shape of the layer
    return input_shape

if __name__ == '__main__':
  symbol = 'BTCUSDT'
  signal_threshold = 0.003
  period = 1
  period_min = '5min'
  seq_length = 60 # 1 hour (12 * 5 minutes)

  prefix_kline = 'kline'
  prefix_orderbook = 'orderbook'
  prefix_long_short_ratio = 'longshortratio'
  prefix_funding_rate = 'fundingrate'
  
  folder_features_kline = f'features/kline/{symbol}/{period}'
  folder_features_orderbook = f'features/orderbook/{symbol}/{period}'
  folder_features_long_short_ratio = f'features/long_short_ratio/{symbol}/{period_min}' 
  folder_features_funding_rate = f'features/funding_rate/{symbol}'

  df_features_kline = pd.read_csv(f'{folder_features_kline}/features.csv')
  df_features_orderbook = pd.read_csv(f'{folder_features_orderbook}/features.csv')
  df_features_long_short_ratio = pd.read_csv(f'{folder_features_long_short_ratio}/features.csv')
  df_features_funding_rate = pd.read_csv(f'{folder_features_funding_rate}/features.csv')

  # Define model output
  df_features_kline['futureRelativePriceChange'] = df_features_kline['relativePriceChange'].shift(-1)

  # Define signal for classification
  df_features_kline['signal'] = 0  # Hold
  df_features_kline.loc[df_features_kline['futureRelativePriceChange'] < -signal_threshold, 'signal'] = 1  # Short
  df_features_kline.loc[df_features_kline['futureRelativePriceChange'] > signal_threshold, 'signal'] = 2  # Long

  # Get min and max timestamp
  timestamp_begin = max(
    df_features_kline['startTime'].min(), 
    df_features_orderbook['timestamp'].min(), 
    df_features_long_short_ratio['timestamp'].min(), 
    df_features_funding_rate['fundingRateTimestamp'].min(),
  )
  timestamp_end = min(
    df_features_kline['startTime'].max(), 
    df_features_orderbook['timestamp'].max(), 
    df_features_long_short_ratio['timestamp'].max(), 
    df_features_funding_rate['fundingRateTimestamp'].max(),
  )

  # Filter dataframes by timestamp_min and timestamp_max
  df_features_kline = df_features_kline[(df_features_kline['startTime'] >= timestamp_begin) & (df_features_kline['startTime'] <= timestamp_end)]
  df_features_orderbook = df_features_orderbook[(df_features_orderbook['timestamp'] >= timestamp_begin) & (df_features_orderbook['timestamp'] <= timestamp_end)]
  df_features_long_short_ratio = df_features_long_short_ratio[(df_features_long_short_ratio['timestamp'] >= timestamp_begin) & (df_features_long_short_ratio['timestamp'] <= timestamp_end)]
  df_features_funding_rate = df_features_funding_rate[(df_features_funding_rate['fundingRateTimestamp'] >= timestamp_begin) & (df_features_funding_rate['fundingRateTimestamp'] <= timestamp_end)]

  # Merge features
  df_features_merged = df_features_kline.copy()
  df_features_merged = merge_features(df_features_merged, df_features_orderbook, 'startTime', 'timestamp', prefix_kline, prefix_orderbook)
  df_features_merged = merge_features(df_features_merged, df_features_long_short_ratio, f'{prefix_kline}_startTime', 'timestamp', None, prefix_long_short_ratio)
  df_features_merged = merge_features(df_features_merged, df_features_funding_rate, f'{prefix_kline}_startTime', 'fundingRateTimestamp', None, prefix_funding_rate)

  # Drop NaN values (last row will be NaN after shifting)
  df_features_merged.dropna(inplace=True)

  # Drop duplicates by f'{prefix_kline}_startTime' column
  df_features_merged = df_features_merged.drop_duplicates(subset=[f'{prefix_kline}_startTime'])

  # Order by timestamp ascending
  df_features_merged = df_features_merged.sort_values(by=f'{prefix_kline}_startTime')

  # Prepare features
  df_features_merged_scaled, scalers_features_kline = prepare_features_kline(df_features_merged, prefix_kline)
  df_features_merged_scaled, scalers_features_orderbook = prepare_features_orderbook(df_features_merged_scaled, prefix_orderbook)
  df_features_merged_scaled, scalers_features_long_short_ratio = prepare_features_long_short_ratio(df_features_merged_scaled, prefix_long_short_ratio)
  df_features_merged_scaled, scalers_features_funding_rate = prepare_features_funding_rate(df_features_merged_scaled, prefix_funding_rate)

  columns_input = []
  columns_input += [f'{prefix_kline}_{col}' for col in columns_input_kline]
  columns_input += [f'{prefix_orderbook}_{col}' for col in columns_input_orderbook]
  columns_input += [f'{prefix_long_short_ratio}_{col}' for col in columns_input_long_short_ratio]
  columns_input += [f'{prefix_funding_rate}_{col}' for col in columns_input_funding_rate]

  # Target variable
  column_target = f'{prefix_kline}_signal'

  X = df_features_merged_scaled[columns_input].values
  y = df_features_merged_scaled[column_target].values  # 0, 1, 2 (Hold, Short, Long)

  # Convert y to one-hot encoding
  y_onehot = tf.keras.utils.to_categorical(y, num_classes=3)

  # Create sequences with seq_length = 60 (5 hours for 5-minute intervals)
  X_seq, y_seq = create_sequences(X, y, seq_length)
  y_seq_onehot = tf.keras.utils.to_categorical(y_seq, num_classes=3)

  # Train-Test Split (80% training, 20% testing)
  X_train_seq, X_test_seq, y_train_seq_onehot, y_test_seq_onehot = train_test_split(
    X_seq, y_seq_onehot, test_size=0.2, shuffle=False
  )

  # Transformer model configuration
  n_features = len(columns_input)  # Number of features (29)
  d_model = n_features  # Set d_model to match input features (29) for consistency
  head_size = 7  # Adjusted to ensure head_size * num_heads fits into d_model (29)
  num_heads = 4  # Number of attention heads, must divide head_size * num_heads evenly into d_model
  ff_dim = 64  # Reduced feed-forward dimension to match d_model for simplicity

  # Define Transformer model
  input_layer = tf.keras.layers.Input(shape=(seq_length, n_features))
  x = TransformerEncoderLayer(head_size, num_heads, ff_dim, dropout=0.1)(input_layer)
  x = tf.keras.layers.GlobalAveragePooling1D()(x)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  x = tf.keras.layers.Dropout(0.1)(x)
  output_layer = tf.keras.layers.Dense(3, activation='softmax')(x)  # 3 classes: Hold, Short, Long

  # Create the model
  model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

  # Compile model
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', 
                metrics=[
                  'accuracy', 
                  tf.keras.metrics.Precision(class_id=0), 
                  tf.keras.metrics.Precision(class_id=1), 
                  tf.keras.metrics.Precision(class_id=2),
                  tf.keras.metrics.Recall(class_id=0), 
                  tf.keras.metrics.Recall(class_id=1), 
                  tf.keras.metrics.Recall(class_id=2),
                  tf.keras.metrics.F1Score(average='weighted')
                ])

  # Class weights for balancing
  class_weights = {
    0: 0.1,  # Hold — минимальный вес
    1: 15.0,  # Short — больший вес
    2: 15.0  # Long — больший вес
  }

  # Train model
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True, verbose=1)
  history = model.fit(X_train_seq, y_train_seq_onehot, epochs=1000, batch_size=32, validation_data=(X_test_seq, y_test_seq_onehot), 
            verbose=1, callbacks=[early_stopping], class_weight=class_weights)

  # Predictions for train and test
  y_train_pred = model.predict(X_train_seq)
  y_test_pred = model.predict(X_test_seq)

  # Convert predictions back to class indices (use the last prediction in sequence for each step)
  y_train_pred_classes = np.argmax(y_train_pred, axis=1)
  y_test_pred_classes = np.argmax(y_test_pred, axis=1)

  print("Train prediction distribution:", np.bincount(y_train_pred_classes))
  print("Test prediction distribution:", np.bincount(y_test_pred_classes))
  print("Train actual distribution:", np.bincount(np.argmax(y_train_seq_onehot, axis=1)))
  print("Test actual distribution:", np.bincount(np.argmax(y_test_seq_onehot, axis=1)))

  # Evaluate metrics
  train_metrics = model.evaluate(X_train_seq, y_train_seq_onehot, verbose=0)
  test_metrics = model.evaluate(X_test_seq, y_test_seq_onehot, verbose=0)

  # Get train and test indices, adjusting for sequence length
  train_indices = df_features_merged.index[seq_length:len(y_train_seq_onehot) + seq_length]
  test_indices = df_features_merged.index[-len(y_test_seq_onehot) - seq_length:-seq_length]
  df_train = df_features_merged.loc[train_indices].copy()
  df_test = df_features_merged.loc[test_indices].copy()

  # Save y_train_pred to file
  pd.DataFrame(y_train_pred).to_csv('predictions/y_train_pred.csv', index=False)
  pd.DataFrame(y_test_pred).to_csv('predictions/y_test_pred.csv', index=False)
  print('y_train_pred and y_test_pred have been saved in CSV format.')

  # Run bot on train and test datasets
  train_profit, train_accuracy = run_bot(df_train, y_train_pred, "Train Dataset", f'{prefix_kline}_closePrice', signal_prob_threshold=0.7)
  test_profit, test_accuracy = run_bot(df_test, y_test_pred, "Test Dataset", f'{prefix_kline}_closePrice', signal_prob_threshold=0.7)

  print("Train prediction distribution:", np.bincount(y_train_pred_classes))
  print("Test prediction distribution:", np.bincount(y_test_pred_classes))
  print("Train actual distribution:", np.bincount(np.argmax(y_train_seq_onehot, axis=1)))
  print("Test actual distribution:", np.bincount(np.argmax(y_test_seq_onehot, axis=1)))

  # Save scaled dataset
  df_features_merged_scaled.to_csv(f'{folder_features_kline}/features_scaled.csv')
  print('\nScaled dataset has been saved.')

  # Save the model and the scaler
  model.save('models/model_kline_transformer.keras')
  print('Model has been saved.')

  # Save the feature and target scalers
  for scaler_name, scaler in scalers_features_kline.items():
    joblib.dump(scaler, f'models/model_kline_scaler_kline_{scaler_name}.pkl')
  for scaler_name, scaler in scalers_features_orderbook.items():
    joblib.dump(scaler, f'models/model_kline_scaler_orderbook_{scaler_name}.pkl')
  for scaler_name, scaler in scalers_features_long_short_ratio.items():
    joblib.dump(scaler, f'models/model_kline_scaler_long_short_ratio_{scaler_name}.pkl')
  for scaler_name, scaler in scalers_features_funding_rate.items():
    joblib.dump(scaler, f'models/model_kline_scaler_funding_rate_{scaler_name}.pkl')
  # for scaler_name, scaler in scalers_features_open_interest.items():
  #   joblib.dump(scaler, f'models/model_kline_scaler_open_interest_{scaler_name}.pkl')
  print('Scalers have been saved.')

  # Original data metrics
  price_changes = [abs(df_features_merged[f'{prefix_kline}_priceChange'])]
  avg_price_change = np.mean(price_changes)
  std_close_price = df_features_merged[f'{prefix_kline}_closePrice'].std()
  avg_candle_range = (df_features_merged[f'{prefix_kline}_highPrice'] - df_features_merged[f'{prefix_kline}_lowPrice']).mean()
  print(f"\nСреднее абсолютное изменение цены (|closePrice[i] - closePrice[i-1]|): {avg_price_change:.2f} USD")
  print(f"Стандартное отклонение closePrice: {std_close_price:.2f} USD")
  print(f"Средний размах свечей (highPrice - lowPrice): {avg_candle_range:.2f} USD")

  # Train metrics
  print(f"\nTrain metrics - "
      f"Accuracy: {train_metrics[1]:.4f}, "
      f"Precision (Hold): {train_metrics[2]:.4f}, Precision (Short): {train_metrics[3]:.4f}, Precision (Long): {train_metrics[4]:.4f}, "
      f"Recall (Hold): {train_metrics[5]:.4f}, Recall (Short): {train_metrics[6]:.4f}, Recall (Long): {train_metrics[7]:.4f}, "
      f"F1 (weighted): {train_metrics[8]:.4f}")
  print(f"Test metrics - "
      f"Accuracy: {test_metrics[1]:.4f}, "
      f"Precision (Hold): {test_metrics[2]:.4f}, Precision (Short): {test_metrics[3]:.4f}, Precision (Long): {test_metrics[4]:.4f}, "
      f"Recall (Hold): {test_metrics[5]:.4f}, Recall (Short): {test_metrics[6]:.4f}, Recall (Long): {test_metrics[7]:.4f}, "
      f"F1 (weighted): {test_metrics[8]:.4f}")

  print("\nTrain predicted distribution:", np.bincount(y_train_pred_classes))
  print("Test predicted distribution:", np.bincount(y_test_pred_classes))

  # Plot actual vs predicted signals for train and test
  plt.figure(figsize=(12, 8))

  plt.subplot(2, 1, 1)
  plt.plot(df_train.index, np.argmax(y_train_seq_onehot, axis=1), label="Actual Signal (Train)", color="blue", alpha=0.7)
  plt.plot(df_train.index, y_train_pred_classes, label="Predicted Signal (Train)", color="red", linestyle="dashed", alpha=0.7)
  plt.xlabel("Time Step")
  plt.ylabel("Signal (0, 1, 2)")
  plt.title("Actual vs Predicted Signals (Train)")
  plt.legend()
  plt.grid(True)

  plt.subplot(2, 1, 2)
  plt.plot(df_test.index, np.argmax(y_test_seq_onehot, axis=1), label="Actual Signal (Test)", color="blue", alpha=0.7)
  plt.plot(df_test.index, y_test_pred_classes, label="Predicted Signal (Test)", color="red", linestyle="dashed", alpha=0.7)
  plt.xlabel("Time Step")
  plt.ylabel("Signal (0, 1, 2)")
  plt.title("Actual vs Predicted Signals (Test)")
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.show()

  # Plot training history
  plt.figure(figsize=(10, 6))
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.plot(history.history['accuracy'], label='Training Accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
  plt.title('Model Loss and Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Loss/Accuracy')
  plt.legend()
  plt.grid(True)
  plt.show()

  plt.hist(df_features_merged[f'{prefix_kline}_futureRelativePriceChange'], bins=50)
  plt.title("Distribution of futureRelativePriceChange")
  plt.show()

  plt.hist(df_features_merged[f'{prefix_kline}_signal'], bins=3)
  plt.title("Distribution of Signals")
  plt.show()
