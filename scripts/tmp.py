import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from utils_broccoli import scale_features
from constants import columns_input_kline

def build_improved_model(timesteps, n_features):
  input_layer = tf.keras.layers.Input(shape=(timesteps, n_features))
  
  # Первый слой LSTM
  layer = tf.keras.layers.LSTM(512, activation='tanh', return_sequences=True)(input_layer)
  layer = tf.keras.layers.BatchNormalization()(layer)
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

def train_model(df_features):
  columns_input = columns_input_kline
  column_target = 'futureClosePrice'

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
  X_train = X_train.astype(np.float32)
  X_test = X_test.astype(np.float32)
  y_train = y_train.astype(np.float32)
  y_test = y_test.astype(np.float32)

  # Build and train model
  model = build_improved_model(timesteps, X_train.shape[2])
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
  history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), 
            verbose=1, callbacks=[early_stopping])

  # Predictions
  y_train_pred = model.predict(X_train)
  y_test_pred = model.predict(X_test)

  # Evaluate metrics
  train_metrics = model.evaluate(X_train, y_train, verbose=0)
  test_metrics = model.evaluate(X_test, y_test, verbose=0)

  print(f'\nTrain metrics - Loss (MSE): {train_metrics[0]:.6f}, MAE: {train_metrics[1]:.6f}')
  print(f'Test metrics - Loss (MSE): {test_metrics[0]:.6f}, MAE: {test_metrics[1]:.6f}')

  # Prepare DataFrames for visualization
  df_train = df_features.iloc[timesteps:timesteps+len(y_train)].copy()
  df_test = df_features.iloc[timesteps+len(y_train):timesteps+len(y_train)+len(y_test)].copy()

  return model, history, df_train, y_train, y_train_pred, df_test, y_test, y_test_pred

if __name__ == '__main__':
  symbol = 'FARTCOINUSDT'
  interval = '5'
  time_start = '2024-01-01'
  amount = 10000
  fee_open = 0.0002
  fee_close = 0.0002

  # Read features
  df_features = pd.read_csv(f'features/kline/{symbol}/{interval}/features.csv')
  df_features['startTime'] = pd.to_datetime(df_features['startTime'], unit='ms')
  df_features.drop_duplicates(subset='startTime', inplace=True)
  df_features = df_features[df_features['startTime'] > pd.to_datetime(time_start)]
  df_features = df_features.sort_values(by='startTime')

  # Define target
  df_features['futureClosePrice'] = df_features['closePrice'].shift(-12)

  # Scale features
  df_features, scalers = scale_features(df_features)

  # Scale target separately to avoid data leakage
  X = df_features[columns_input_kline].values
  y = df_features['futureClosePrice'].values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
  scalers['output'] = MinMaxScaler(feature_range=(0, 1))
  y_train = scalers['output'].fit_transform(y_train.reshape(-1, 1)).flatten()
  y_test = scalers['output'].transform(y_test.reshape(-1, 1)).flatten()
  df_features['futureClosePrice'] = np.concatenate([y_train, y_test])

  # Remove NaN
  df_features.dropna(inplace=True)

  # Train model
  model, history, df_train, y_train, y_train_pred, df_test, y_test, y_test_pred = train_model(df_features)

  # Save model and scalers
  model.save(f'models/model_broccoli_{symbol}.keras')
  for scaler_name, scaler in scalers.items():
    joblib.dump(scaler, f'models/model_broccoli_scaler_{symbol}_{scaler_name}.pkl')

  df_train.to_csv(f'models/model_broccoli_train_{symbol}.csv')
  df_test.to_csv(f'models/model_broccoli_test_{symbol}.csv')
  np.save(f'models/model_broccoli_train_y_pred_{symbol}.npy', y_train_pred)
  np.save(f'models/model_broccoli_test_y_pred_{symbol}.npy', y_test_pred)

  # Plot actual vs predicted
  plt.figure(figsize=(12, 8))
  plt.subplot(2, 1, 1)
  plt.plot(df_train['startTime'], y_train, label='Actual Future Price (Train)', color='blue', alpha=0.7)
  plt.plot(df_train['startTime'], y_train_pred, label='Predicted Future Price (Train)', color='red', linestyle='dashed', alpha=0.7)
  plt.xlabel('Datetime')
  plt.ylabel('Future Close Price')
  plt.title('Actual vs Predicted Future Prices (Train)')
  plt.legend()
  plt.grid(True)

  plt.subplot(2, 1, 2)
  plt.plot(df_test['startTime'], y_test, label='Actual Future Price (Test)', color='blue', alpha=0.7)
  plt.plot(df_test['startTime'], y_test_pred, label='Predicted Future Price (Test)', color='red', linestyle='dashed', alpha=0.7)
  plt.xlabel('Datetime')
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