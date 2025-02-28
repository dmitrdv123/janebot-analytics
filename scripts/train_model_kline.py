import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

def prepare_features_kline(df):
  df_scaled = df.copy()

  # Define columns for scale 
  target_columns = ['futureClosePrice']
  price_columns = ['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10']
  price_change_column = ['priceChange']
  volume_columns = ['volume']
  turnover_column = ['turnover']
  returns_columns = ['logReturn', 'stdReturn_5m', 'stdReturn_10m', 'MACD_line', 'MACD_signal', 'MACD_histogram', 
             'Stochastic_K', 'Stochastic_D', 'ROC_14', 'RSI_14']
  range_columns = ['highLowRange']

  # Initialize scalers
  scalers = {
    'target': MinMaxScaler(),
    'price': MinMaxScaler(),
    'price_change': RobustScaler(),
    'volume': RobustScaler(),
    'turnover': StandardScaler(),
    'returns': StandardScaler(),
    'range': MinMaxScaler()
  }

  # Apply different scalers
  df_scaled[target_columns] = scalers['target'].fit_transform(df_scaled[target_columns])
  df_scaled[price_columns] = scalers['price'].fit_transform(df_scaled[price_columns])
  df_scaled[price_change_column] = scalers['price_change'].fit_transform(df_scaled[price_change_column])
  df_scaled[volume_columns] = scalers['volume'].fit_transform(df_scaled[volume_columns])
  df_scaled[returns_columns] = scalers['returns'].fit_transform(df_scaled[returns_columns])
  df_scaled[range_columns] = scalers['range'].fit_transform(df_scaled[range_columns])

  # Apply log transformation to turnover before scaling
  df_scaled[turnover_column] = np.log1p(df_scaled[turnover_column])  # log1p to avoid log(0) issues
  df_scaled[turnover_column] = scalers['turnover'].fit_transform(df_scaled[turnover_column])

  # Apply scale to categorical features
  df_scaled['hourOfDay'] = df_scaled['hourOfDay'] / 23  # Normalize to [0,1]
  df_scaled['dayOfWeek'] = df_scaled['dayOfWeek'] / 6  # Normalize to [0,1] (0=Monday, 6=Sunday)
  df_scaled['weekOfYear'] = df_scaled['weekOfYear'] / 51 # Normalize to [0,1]
  df_scaled['monthOfYear'] = df_scaled['monthOfYear'] / 11 # Normalize to [0,1] (0=Jan, 11=Dec)
  df_scaled['minuteOfHour'] = df_scaled['minuteOfHour'] / 59 # Normalize to [0,1]

  return df_scaled, scalers

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

  # Prepare features
  df_features_kline_scaled, scalers_features_kline = prepare_features_kline(df_features_kline)

  # Define feature columns
  input_columns = [
    'openPrice', 
    'highPrice', 
    'lowPrice', 
    'closePrice', 
    'volume', 
    'turnover', 
    'priceChange', 
    'relativePriceChange', 
    'logReturn', 
    'SMA_5', 
    'SMA_10', 
    'EMA_5', 
    'EMA_10',
    'hourOfDay', 
    'dayOfWeek', 
    'weekOfYear', 
    'monthOfYear', 
    'minuteOfHour', 
    'isWeekend', 
    'highLowRange', 
    'stdReturn_5m', 
    'stdReturn_10m', 
    'RSI_14', 
    'MACD_line', 
    'MACD_signal', 
    'MACD_histogram', 
    'Stochastic_K', 
    'Stochastic_D', 
    'ROC_14',
  ]

  # Target variable (future relative price change)
  target_column = 'futureClosePrice'

  # Prepare input and output data
  X = df_features_kline_scaled[input_columns].values # Features
  y = df_features_kline_scaled[target_column].values # Target (future relative price change)

  # Train-Test Split (80% training, 20% testing)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

  # Reshape data for LSTM (samples, time steps, features)
  X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
  X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

  # Define LSTM model
  input_layer = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
  layer = tf.keras.layers.LSTM(128, activation='tanh', return_sequences=False)(input_layer)
  layer = tf.keras.layers.LeakyReLU()(layer)
  layer = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01))(layer)
  layer = tf.keras.layers.LeakyReLU()(layer)
  layer = tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01))(layer)
  layer = tf.keras.layers.LeakyReLU()(layer)
  output_layer = tf.keras.layers.Dense(1)(layer)

  # Create the model
  model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

  # Compile model
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(optimizer=optimizer, loss='mean_squared_error')

  # Train model
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
  model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

  # Predictions
  y_pred = model.predict(X_test)

  # Unscale y_pred and y_test
  y_pred_unscaled = scalers_features_kline['target'].inverse_transform(y_pred.reshape(-1, 1)).flatten()
  y_test_unscaled = scalers_features_kline['target'].inverse_transform(y_test.reshape(-1, 1)).flatten()

  # Calculate Errors on unscaled data
  mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
  mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)

  print(f'Mean Squared Error (MSE): {mse:.6f}')
  print(f'Mean Absolute Error (MAE): {mae:.6f}')

  # Get test indices to align with original DataFrame
  test_indices = df_features_kline.index[-len(y_test_unscaled):]
  df_test = df_features_kline.loc[test_indices].copy()
  df_test['predictedClosePrice'] = y_pred_unscaled

  # Calculate metrics on original data
  price_changes = [abs(df_features_kline["priceChange"])]
  avg_price_change = np.mean(price_changes)
  print(f"Среднее абсолютное изменение цены (|closePrice[i] - closePrice[i-1]|): {avg_price_change:.2f} USD")

  std_close_price = df_features_kline["closePrice"].std()
  print(f"Стандартное отклонение closePrice: {std_close_price:.2f} USD")

  avg_candle_range = (df_features_kline["highPrice"] - df_features_kline["lowPrice"]).mean()
  print(f"Средний размах свечей (highPrice - lowPrice): {avg_candle_range:.2f} USD")

  # Bot logic
  fee_open = 0.0002
  fee_close = 0.0002
  rmse_threshold = 70
  correlator_threshold = 0.7
  volatility_threshold = 50
  long_signals = 0
  short_signals = 0
  long_correct = 0
  short_correct = 0
  total_profit = 0

  for i in range(len(df_test) - 1):  # -1 to avoid predicting last row without next price
    current_row = df_test.iloc[i]
    next_row = df_test.iloc[i + 1]
    current_close = current_row['closePrice']
    predicted_close = current_row['predictedClosePrice']
    expected_change = abs(predicted_close - current_close)

    # Calculate trend_strength and mean_correlator (simplified approximation)
    lookback = min(1000, i)
    hist_df = df_test.iloc[max(0, i - lookback):i + 1]
    trend_strength = np.corrcoef(hist_df['startTime'], hist_df['closePrice'])[0, 1] if len(hist_df) > 1 else 0
    mean_correlator = abs(trend_strength)  # Simplified proxy for correlator

    # Volatility filter
    vol_lookback = min(10, i)
    recent_volatility = (df_test['highPrice'].iloc[max(0, i - vol_lookback):i + 1] - 
               df_test['lowPrice'].iloc[max(0, i - vol_lookback):i + 1]).mean()

    # Trading signals
    signal = None
    if expected_change > rmse_threshold and mean_correlator > correlator_threshold and recent_volatility > volatility_threshold:
      if predicted_close > current_close:
        signal = "Long"
      elif predicted_close < current_close and trend_strength < 0:
        signal = "Short"

    # Analyze trading signals
    if signal == "Long":
      long_signals += 1
      next_price = next_row['closePrice']
      profit = next_price - current_close - fee_open * current_close - fee_close * next_price
      stop_loss = max(-0.5 * expected_change, -30)
      if profit < stop_loss:
        profit = stop_loss - fee_open * current_close - fee_close * next_price
      total_profit += profit
      if next_price > current_close:
        long_correct += 1
    elif signal == "Short":
      short_signals += 1
      next_price = next_row['closePrice']
      profit = current_close - next_price - fee_open * current_close - fee_close * next_price
      stop_loss = max(-0.5 * expected_change, -30)
      if profit < stop_loss:
        profit = stop_loss - fee_open * current_close - fee_close * next_price
      total_profit += profit
      if next_price < current_close:
        short_correct += 1

  # Bot results
  total_signals = long_signals + short_signals
  print(f"\nРезультаты бота:")
  print(f"Всего сигналов: {total_signals}")
  print(f"Сигналов Long: {long_signals}")
  print(f"Точных Long: {long_correct} ({long_correct/long_signals*100:.2f}% если >0)" if long_signals > 0 else "Точных Long: 0 (0%)")
  print(f"Сигналов Short: {short_signals}")
  print(f"Точных Short: {short_correct} ({short_correct/short_signals*100:.2f}% если >0)" if short_signals > 0 else "Точных Short: 0 (0%)")
  total_correct = long_correct + short_correct
  accuracy = total_correct / total_signals * 100 if total_signals > 0 else 0
  print(f"Общая точность: {accuracy:.2f}%")
  print(f"Общая прибыль/убыток (с адаптивным стоп-лоссом, без тейк-профита, без комиссий): {total_profit:.2f} USD")
  avg_profit_per_trade = total_profit / total_signals if total_signals > 0 else 0
  print(f"Средняя прибыль на сделку: {avg_profit_per_trade:.2f} USD")

  # Plot actual vs predicted close price
  plt.figure(figsize=(12, 6))
  plt.plot(y_test_unscaled, label="Actual Close Price", color="blue", alpha=0.7)
  plt.plot(y_pred_unscaled, label="Predicted Close Price", color="red", linestyle="dashed", alpha=0.7)
  plt.xlabel("Time Step")
  plt.ylabel("Close Price (USD)")
  plt.title("Actual vs. Predicted Close Price")
  plt.legend()
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
