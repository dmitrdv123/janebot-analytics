import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

def prepare_features_kline(df):
  df_scaled = df.copy()

  # Define columns for scale 
  price_columns = ['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10']
  price_change_column = ['priceChange']
  relative_price_change_column = ['relativePriceChange']
  volume_columns = ['volume']
  turnover_column = ['turnover']
  returns_columns = ['logReturn', 'stdReturn_5m', 'stdReturn_10m', 'MACD_line', 'MACD_signal', 'MACD_histogram', 
             'Stochastic_K', 'Stochastic_D', 'ROC_14', 'RSI_14']
  range_columns = ['highLowRange']

  # Initialize scalers
  scalers = {
    'price': MinMaxScaler(),
    'price_change': RobustScaler(),
    'relative_price_change': MinMaxScaler(feature_range=(-1, 1)),
    'volume': RobustScaler(),
    'turnover': StandardScaler(),
    'returns': StandardScaler(),
    'range': MinMaxScaler()
  }

  # Apply different scalers
  df_scaled[price_columns] = scalers['price'].fit_transform(df_scaled[price_columns])
  df_scaled[price_change_column] = scalers['price_change'].fit_transform(df_scaled[price_change_column])
  df_scaled[relative_price_change_column] = scalers['relative_price_change'].fit_transform(df_scaled[relative_price_change_column])
  df_scaled[volume_columns] = scalers['volume'].fit_transform(df_scaled[volume_columns])
  df_scaled[returns_columns] = scalers['returns'].fit_transform(df_scaled[returns_columns])
  df_scaled[range_columns] = scalers['range'].fit_transform(df_scaled[range_columns])

  # Apply log transformation to turnover before scaling
  df_scaled[turnover_column] = np.log1p(df_scaled[turnover_column])  # log1p to avoid log(0) issues
  df_scaled[turnover_column] = scalers['turnover'].fit_transform(df_scaled[turnover_column])

  # Apply scale to categorical features
  df_scaled['hourOfDay'] = df_scaled['hourOfDay'] / 23  # Normalize to [0,1]
  df_scaled['dayOfWeek'] = df_scaled['dayOfWeek'] / 6  # Normalize to [0,1] (0=Monday, 6=Sunday)
  df_scaled['weekOfYear'] = df_scaled['weekOfYear'] / 51  # Normalize to [0,1]
  df_scaled['monthOfYear'] = df_scaled['monthOfYear'] / 11  # Normalize to [0,1] (0=Jan, 11=Dec)
  df_scaled['minuteOfHour'] = df_scaled['minuteOfHour'] / 59  # Normalize to [0,1]

  return df_scaled, scalers

def run_bot(df, y_pred, dataset_name):
  position_amount_to_use = 100000
  fee_open = 0.0002
  fee_close = 0.0002
  long_signals = 0
  short_signals = 0
  long_correct = 0
  short_correct = 0
  total_profit = 0

  for i in range(len(df) - 1):
    current_row = df.iloc[i]
    next_row = df.iloc[i + 1]
    current_close = current_row['closePrice']
    signal = np.argmax(y_pred[i])  # 0 (Hold), 1 (Short), 2 (Long)

    if signal == 2:  # Long
      long_signals += 1
      next_price = next_row['closePrice']

      position_amount = position_amount_to_use
      gross_profit = (next_price - current_close) / current_close * position_amount
      next_position_amount = position_amount + gross_profit
      open_fee = position_amount * fee_open
      close_fee = next_position_amount * fee_close
      profit = gross_profit - open_fee - close_fee

      stop_loss = -30
      if profit < stop_loss:
        profit = stop_loss
      total_profit += profit
      if next_price > current_close:
        long_correct += 1
    elif signal == 1:  # Short
      short_signals += 1
      next_price = next_row['closePrice']

      position_amount = position_amount_to_use
      gross_profit = (current_close - next_price) / current_close * position_amount
      next_position_amount = position_amount + gross_profit
      open_fee = position_amount * fee_open
      close_fee = next_position_amount * fee_close
      profit = gross_profit - open_fee - close_fee

      stop_loss = -30
      if profit < stop_loss:
        profit = stop_loss - open_fee - close_fee
      total_profit += profit
      if next_price < current_close:
        short_correct += 1

  # Bot results
  total_signals = long_signals + short_signals
  total_correct = long_correct + short_correct
  accuracy = total_correct / total_signals * 100 if total_signals > 0 else 0
  avg_profit_per_trade = total_profit / total_signals if total_signals > 0 else 0

  print(f"\nРезультаты бота для {dataset_name}:")
  print(f"Всего сигналов: {total_signals}")
  print(f"Сигналов Long: {long_signals}")
  print(f"Точных Long: {long_correct} ({long_correct/long_signals*100:.2f}% если >0)" if long_signals > 0 else "Точных Long: 0 (0%)")
  print(f"Сигналов Short: {short_signals}")
  print(f"Точных Short: {short_correct} ({short_correct/short_signals*100:.2f}% если >0)" if short_signals > 0 else "Точных Short: 0 (0%)")
  print(f"Общая точность: {accuracy:.2f}%")
  print(f"Общая прибыль/убыток: {total_profit:.2f} USD")
  print(f"Средняя прибыль на сделку: {avg_profit_per_trade:.2f} USD")

  return total_profit, accuracy

if __name__ == '__main__':
  symbol = 'BTCUSDT'
  folder_features_kline = f'features/kline/{symbol}/5'

  df_features_kline = pd.read_csv(f'{folder_features_kline}/features.csv')

  # Define model output
  df_features_kline['futureRelativePriceChange'] = df_features_kline['relativePriceChange'].shift(-1)
  df_features_kline['futurePriceChange'] = df_features_kline['priceChange'].shift(-1)
  df_features_kline['futureClosePrice'] = df_features_kline['closePrice'].shift(-1)
  
  # Order by timestamp ascending
  df_features_kline = df_features_kline.sort_values(by='startTime')
  
  # Drop NaN values (last row will be NaN after shifting)
  df_features_kline.dropna(inplace=True)

  # Define signal for classification
  df_features_kline['signal'] = 0  # Hold
  df_features_kline.loc[df_features_kline['futureRelativePriceChange'] < -0.003, 'signal'] = 1  # Short
  df_features_kline.loc[df_features_kline['futureRelativePriceChange'] > 0.003, 'signal'] = 2  # Long

  # Prepare features
  df_features_kline_scaled, scalers_features_kline = prepare_features_kline(df_features_kline)

  # Define feature columns
  input_columns = [
    'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover',
    'priceChange', 'relativePriceChange', 'logReturn', 'SMA_5', 'SMA_10',
    'EMA_5', 'EMA_10', 'hourOfDay', 'dayOfWeek', 'weekOfYear', 'monthOfYear',
    'minuteOfHour', 'isWeekend', 'highLowRange', 'stdReturn_5m', 'stdReturn_10m',
    'RSI_14', 'MACD_line', 'MACD_signal', 'MACD_histogram', 'Stochastic_K',
    'Stochastic_D', 'ROC_14'
  ]

  # Target variable
  target_column = 'signal'

  # Prepare input and output data
  X = df_features_kline_scaled[input_columns].values
  y = df_features_kline_scaled[target_column].values  # 0, 1, 2 (Hold, Short, Long)

  # Convert y to one-hot encoding
  y_onehot = tf.keras.utils.to_categorical(y, num_classes=3)

  # Train-Test Split (80% training, 20% testing)
  X_train, X_test, y_train_onehot, y_test_onehot = train_test_split(X, y_onehot, test_size=0.2, shuffle=False)

  # Reshape data for LSTM
  X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
  X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

  # Define LSTM model
  input_layer = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
  layer = tf.keras.layers.LSTM(256, activation='tanh', return_sequences=True)(input_layer)
  layer = tf.keras.layers.Dropout(0.1)(layer)  # Уменьшаем Dropout
  layer = tf.keras.layers.LSTM(128, activation='tanh', return_sequences=False)(layer)
  layer = tf.keras.layers.Dropout(0.1)(layer)
  layer = tf.keras.layers.LeakyReLU()(layer)
  layer = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01))(layer)  # Увеличиваем регуляризацию
  layer = tf.keras.layers.LeakyReLU()(layer)
  layer = tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01))(layer)
  layer = tf.keras.layers.LeakyReLU()(layer)
  output_layer = tf.keras.layers.Dense(3, activation='softmax')(layer)  # 3 класса: 0 (Hold), 1 (Short), 2 (Long)

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
  history = model.fit(X_train, y_train_onehot, epochs=1000, batch_size=32, validation_data=(X_test, y_test_onehot), 
            verbose=1, callbacks=[early_stopping], class_weight=class_weights)

  # Predictions for train and test
  y_train_pred = model.predict(X_train)
  y_test_pred = model.predict(X_test)

  # Convert predictions back to class indices
  y_train_pred_classes = np.argmax(y_train_pred, axis=1)
  y_test_pred_classes = np.argmax(y_test_pred, axis=1)

  # Evaluate metrics
  train_metrics = model.evaluate(X_train, y_train_onehot, verbose=0)
  test_metrics = model.evaluate(X_test, y_test_onehot, verbose=0)

  # Print metrics for all classes
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

  # Get train and test indices to align with original DataFrame
  train_indices = df_features_kline.index[:len(y_train_onehot)]
  test_indices = df_features_kline.index[-len(y_test_onehot):]
  df_train = df_features_kline.loc[train_indices].copy()
  df_test = df_features_kline.loc[test_indices].copy()

  # Run bot on train and test datasets
  train_profit, train_accuracy = run_bot(df_train, y_train_pred, "Train Dataset")
  test_profit, test_accuracy = run_bot(df_test, y_test_pred, "Test Dataset")

  # Plot actual vs predicted signals for train and test
  plt.figure(figsize=(12, 8))

  plt.subplot(2, 1, 1)
  plt.plot(df_train.index, np.argmax(y_train_onehot, axis=1), label="Actual Signal (Train)", color="blue", alpha=0.7)
  plt.plot(df_train.index, y_train_pred_classes, label="Predicted Signal (Train)", color="red", linestyle="dashed", alpha=0.7)
  plt.xlabel("Time Step")
  plt.ylabel("Signal (0, 1, 2)")
  plt.title("Actual vs Predicted Signals (Train)")
  plt.legend()
  plt.grid(True)
  
  print("Train predicted distribution:", np.bincount(y_train_pred_classes))
  print("Test predicted distribution:", np.bincount(y_test_pred_classes))

  plt.subplot(2, 1, 2)
  plt.plot(df_test.index, np.argmax(y_test_onehot, axis=1), label="Actual Signal (Test)", color="blue", alpha=0.7)
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

  plt.hist(df_features_kline['signal'], bins=3)
  plt.title("Distribution of Signals")
  plt.show()

  # Save scaled dataset
  df_features_kline_scaled.to_csv(f'{folder_features_kline}/features_scaled.csv')
  print('Scaled dataset has been saved.')

  # Save the model and the scaler
  model.save('models/model_kline_lstm.keras')
  print('Model has been saved.')

  # Save the feature and target scalers
  for scaler_name, scaler in scalers_features_kline.items():
    joblib.dump(scaler, f'models/model_kline_scaler_{scaler_name}.pkl')
  print('Scalers have been saved.')