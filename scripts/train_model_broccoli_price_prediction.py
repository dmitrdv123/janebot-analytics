import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable

from utils import ensure_directory
from utils_broccoli import scale_features
from constants import columns_input_kline

@register_keras_serializable()
class Attention(Layer):
  def __init__(self):
    super(Attention, self).__init__()

  def build(self, input_shape):
    self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                             initializer='random_normal', trainable=True)
    self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                             initializer='zeros', trainable=True)
    super(Attention, self).build(input_shape)

  def call(self, inputs):
    # Вычисляем веса внимания
    e = K.tanh(K.dot(inputs, self.W) + self.b)
    e = K.squeeze(e, axis=-1)
    alpha = K.softmax(e)
    alpha = K.expand_dims(alpha, axis=-1)
    # Взвешиваем входные данные
    context = inputs * alpha
    context = K.sum(context, axis=1)
    return context

  def get_config(self):
    config = super(Attention, self).get_config()
    return config

def build_attention_model(timesteps, n_features):
  input_layer = tf.keras.layers.Input(shape=(timesteps, n_features))
  
  # LSTM слои
  layer = tf.keras.layers.LSTM(512, activation='tanh', return_sequences=True)(input_layer)
  layer = tf.keras.layers.BatchNormalization()(layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  
  layer = tf.keras.layers.LSTM(256, activation='tanh', return_sequences=True)(layer)
  layer = tf.keras.layers.BatchNormalization()(layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  
  # Добавляем Attention
  layer = Attention()(layer)
  
  # Полносвязные слои
  layer = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(layer)
  layer = tf.keras.layers.BatchNormalization()(layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  
  layer = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(layer)
  layer = tf.keras.layers.BatchNormalization()(layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  
  output_layer = tf.keras.layers.Dense(1)(layer)
  
  model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
  
  return model

def build_bidirectional_model(timesteps, n_features):
  input_layer = tf.keras.layers.Input(shape=(timesteps, n_features))
  
  # Bidirectional LSTM
  layer = tf.keras.layers.Bidirectional(
      tf.keras.layers.LSTM(512, activation='tanh', return_sequences=True)
  )(input_layer)
  layer = tf.keras.layers.BatchNormalization()(layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  
  layer = tf.keras.layers.Bidirectional(
      tf.keras.layers.LSTM(256, activation='tanh', return_sequences=False)
  )(layer)
  layer = tf.keras.layers.BatchNormalization()(layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  
  # Полносвязные слои
  layer = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(layer)
  layer = tf.keras.layers.BatchNormalization()(layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  
  layer = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(layer)
  layer = tf.keras.layers.BatchNormalization()(layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  
  output_layer = tf.keras.layers.Dense(1)(layer)
  
  model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
  model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
  
  return model

def build_improved_model(timesteps, n_features):
  input_layer = tf.keras.layers.Input(shape=(timesteps, n_features))
  
  # Первый слой LSTM с возвратом последовательностей
  layer = tf.keras.layers.LSTM(512, activation='tanh', return_sequences=True)(input_layer)
  layer = tf.keras.layers.BatchNormalization()(layer)  # Стабилизация обучения
  layer = tf.keras.layers.Dropout(0.2)(layer)
  
  # Второй слой LSTM
  layer = tf.keras.layers.LSTM(256, activation='tanh', return_sequences=True)(layer)
  layer = tf.keras.layers.BatchNormalization()(layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  
  # Третий слой LSTM
  layer = tf.keras.layers.LSTM(128, activation='tanh', return_sequences=False)(layer)
  layer = tf.keras.layers.BatchNormalization()(layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  
  # Полносвязные слои
  layer = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(layer)
  layer = tf.keras.layers.BatchNormalization()(layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  
  layer = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(layer)
  layer = tf.keras.layers.BatchNormalization()(layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  
  output_layer = tf.keras.layers.Dense(1)(layer)  # Линейная активация для регрессии
  
  model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Уменьшенный learning rate
  model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
  
  return model

def build_model(timesteps, n_features):
  input_layer = tf.keras.layers.Input(shape=(timesteps, n_features))
  layer = tf.keras.layers.LSTM(256, activation='tanh', return_sequences=True)(input_layer)
  layer = tf.keras.layers.Dropout(0.1)(layer)
  layer = tf.keras.layers.LSTM(128, activation='tanh', return_sequences=False)(layer)
  layer = tf.keras.layers.Dropout(0.1)(layer)
  layer = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(layer)
  layer = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(layer)
  output_layer = tf.keras.layers.Dense(1)(layer)  # Линейная активация для регрессии

  # Create the model
  model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

  # Compile model
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])  # Mean Absolute Error как метрика
  
  return model

def train_model(df_features):
  # Define feature columns
  columns_input = columns_input_kline

  # Target variable
  column_target = 'futureClosePrice'

  # Prepare input and output data
  X = df_features[columns_input].values
  y = df_features[column_target].values

  # Создаем последовательности из нескольких временных шагов
  timesteps = 5
  X_seq = []
  y_seq = []
  for i in range(timesteps, len(X)):
    X_seq.append(X[i-timesteps:i])
    y_seq.append(y[i])
  X_seq = np.array(X_seq)
  y_seq = np.array(y_seq)

  # Train-Test Split
  X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

  # Приводим к float32 для TensorFlow
  X_train = X_train.astype(np.float32)
  X_test = X_test.astype(np.float32)
  y_train = y_train.astype(np.float32)
  y_test = y_test.astype(np.float32)

  # Create the model
  model = build_attention_model(timesteps, X_train.shape[2])

  # Train model
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
  history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), 
                      verbose=1, callbacks=[early_stopping])

  # Predictions for train and test
  y_train_pred = model.predict(X_train)
  y_test_pred = model.predict(X_test)

  # Evaluate metrics
  train_metrics = model.evaluate(X_train, y_train, verbose=0)
  test_metrics = model.evaluate(X_test, y_test, verbose=0)

  # Print metrics
  print(f'\nTrain metrics - Loss (MSE): {train_metrics[0]:.6f}, MAE: {train_metrics[1]:.6f}')
  print(f'Test metrics - Loss (MSE): {test_metrics[0]:.6f}, MAE: {test_metrics[1]:.6f}')

  # Unscale the predictions and actual values
  y_train_unscaled = scalers['output'].inverse_transform(y_train.reshape(-1, 1)).flatten()
  y_train_pred_unscaled = scalers['output'].inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
  y_test_unscaled = scalers['output'].inverse_transform(y_test.reshape(-1, 1)).flatten()
  y_test_pred_unscaled = scalers['output'].inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

  # Compute percentage errors (MAPE and MSPE)
  # Avoid division by zero by adding a small epsilon to the denominator
  epsilon = 1e-10
  train_mape = np.mean(np.abs((y_train_unscaled - y_train_pred_unscaled) / (y_train_unscaled + epsilon))) * 100
  test_mape = np.mean(np.abs((y_test_unscaled - y_test_pred_unscaled) / (y_test_unscaled + epsilon))) * 100

  train_mspe = np.mean(((y_train_unscaled - y_train_pred_unscaled) / (y_train_unscaled + epsilon)) ** 2) * 100
  test_mspe = np.mean(((y_test_unscaled - y_test_pred_unscaled) / (y_test_unscaled + epsilon)) ** 2) * 100

  print(f'\nPercentage Errors:')
  print(f'Train MAPE: {train_mape:.2f}%')
  print(f'Test MAPE: {test_mape:.2f}%')
  print(f'Train MSPE: {train_mspe:.2f}%')
  print(f'Test MSPE: {test_mspe:.2f}%')

  # Compute directional accuracy
  y_train_direction = np.sign(y_train[1:] - y_train[:-1])
  y_train_pred_direction = np.sign(y_train_pred[1:] - y_train_pred[:-1])
  y_test_direction = np.sign(y_test[1:] - y_test[:-1])
  y_test_pred_direction = np.sign(y_test_pred[1:] - y_test_pred[:-1])

  train_directional_accuracy = np.mean(y_train_direction == y_train_pred_direction)
  test_directional_accuracy = np.mean(y_test_direction == y_test_pred_direction)

  print(f'Train Directional Accuracy: {train_directional_accuracy:.4f}')
  print(f'Test Directional Accuracy: {test_directional_accuracy:.4f}')

  # Возвращаем DataFrame с предсказаниями для удобства визуализации
  df_train = df_features.iloc[:len(y_train)].copy()
  df_test = df_features.iloc[len(y_train):len(y_train) + len(y_test)].copy()

  return model, history, df_train, y_train, y_train_pred, df_test, y_test, y_test_pred

if __name__ == '__main__':
  symbol = 'BROCCOLIUSDT'
  # symbol = 'MELANIAUSDT'
  # symbol = 'TRUMPUSDT'
  # symbol = 'FARTCOINUSDT'
  # symbol = 'BTCUSDT'
  interval = '5'  # Kline interval (1m, 5m, 15m, etc.)
  time_start = '2024-01-01'
  amount = 10000
  model_name = 'attention'

  fee_open = 0.0002
  fee_close = 0.0002

  # Read features
  df_features = pd.read_csv(f'features/kline/{symbol}/{interval}/features.csv')

  # Convert timestamp to datetime
  df_features['startTime'] = pd.to_datetime(df_features['startTime'], unit='ms')

  # Drop duplicates
  df_features.drop_duplicates(subset='startTime', inplace=True)

  # Filter by timestamp
  df_features = df_features[df_features['startTime'] > pd.to_datetime(time_start)]

  # Order by timestamp ascending
  df_features = df_features.sort_values(by='startTime')

  # Define model output
  df_features['futureClosePrice'] = df_features['closePrice'].shift(-12)

  # Scale features
  df_features, scalers = scale_features(df_features)

  scalers['output'] = StandardScaler()
  df_features[['futureClosePrice']] = scalers['output'].fit_transform(df_features[['futureClosePrice']])

  # Remove NaN
  df_features.dropna(inplace=True)

  model, history, df_train, y_train, y_train_pred, df_test, y_test, y_test_pred = train_model(df_features)

  # Create dir for saving
  dir = f'models/{symbol}/{model_name}'
  ensure_directory(dir)

  # Save model
  model.save(f'{dir}/model.keras')

  # Save scalers
  for scaler_name, scaler in scalers.items():
    joblib.dump(scaler, f'{dir}/scaler_{scaler_name}.pkl')

  df_train.to_csv(f'{dir}/train.csv')
  df_test.to_csv(f'{dir}/test.csv')
  np.save(f'{dir}/train_y_pred.npy', y_train_pred)
  np.save(f'{dir}/test_y_pred.npy', y_test_pred)

  # Plot actual vs predicted prices for train and test
  plt.figure(figsize=(12, 8))

  # Сдвигаем временные метки на 12 шагов вперед
  plt.plot(df_train['startTime'], y_train, label='Actual Future Price (Train)', color='blue', alpha=0.7)
  plt.plot(df_train['startTime'], y_train_pred, label='Predicted Future Price (Train)', color='red', linestyle='dashed', alpha=0.7)
  plt.xlabel('Time Step')
  plt.ylabel('Future Close Price')
  plt.title('Actual vs Predicted Future Prices (Train)')
  plt.legend()
  plt.grid(True)

  # Test Plot
  plt.subplot(2, 1, 2)
  # Сдвигаем временные метки на 12 шагов вперед
  plt.plot(df_test['startTime'], y_test, label='Actual Future Price (Test)', color='blue', alpha=0.7)
  plt.plot(df_test['startTime'], y_test_pred, label='Predicted Future Price (Test)', color='red', linestyle='dashed', alpha=0.7)
  plt.xlabel('Time Step')
  plt.ylabel('Future Close Price')
  plt.title('Actual vs Predicted Future Prices (Test)')
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.show()

  # Plot training history
  plt.figure(figsize=(10, 6))
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.plot(history.history['mae'], label='Training MAE')
  plt.plot(history.history['val_mae'], label='Validation MAE')
  plt.title('Model Loss and MAE')
  plt.xlabel('Epoch')
  plt.ylabel('Loss/MAE')
  plt.legend()
  plt.grid(True)
  plt.show()
