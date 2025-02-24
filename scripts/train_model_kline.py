import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

def prepare_features_kline(df):
  # Define columns for scale 
  price_columns = ['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10']
  price_change_column = ['priceChange']
  volume_columns = ['volume']
  turnover_column = ['turnover']
  returns_columns = ['logReturn', 'stdReturn_5m', 'stdReturn_10m', 'MACD_line', 'MACD_signal', 'MACD_histogram', 
             'Stochastic_K', 'Stochastic_D', 'ROC_14', 'RSI_14']
  range_columns = ['highLowRange']

  # Initialize scalers
  scalers = {
    'price': MinMaxScaler(),
    'price_change': RobustScaler(),
    'volume': RobustScaler(),
    'turnover': StandardScaler(),
    'returns': StandardScaler(),
    'range': MinMaxScaler()
  }

  # Apply different scalers
  df[price_columns] = scalers['price'].fit_transform(df[price_columns])
  df[price_change_column] = scalers['price_change'].fit_transform(df[price_change_column])
  df[volume_columns] = scalers['volume'].fit_transform(df[volume_columns])
  df[returns_columns] = scalers['returns'].fit_transform(df[returns_columns])
  df[range_columns] = scalers['range'].fit_transform(df[range_columns])

  # Apply log transformation to turnover before scaling
  df[turnover_column] = np.log1p(df[turnover_column])  # log1p to avoid log(0) issues
  df[turnover_column] = scalers['turnover'].fit_transform(df[turnover_column])

  # Apply scale to categorical features
  df['hourOfDay'] = df['hourOfDay'] / 23  # Normalize to [0,1]
  df['dayOfWeek'] = df['dayOfWeek'] / 6    # Normalize to [0,1] (0=Monday, 6=Sunday)
  df['weekOfYear'] = df['weekOfYear'] / 51 # Normalize to [0,1]
  df['monthOfYear'] = df['monthOfYear'] / 11 # Normalize to [0,1] (0=Jan, 11=Dec)
  df['minuteOfHour'] = df['minuteOfHour'] / 59 # Normalize to [0,1]

  # Define model output  
  df['futureRelativePriceChange'] = df['relativePriceChange'].shift(-1)
  df['futurePriceChange'] = df['priceChange'].shift(-1)
  df['futureClosePrice'] = df['closePrice'].shift(-1)

  # Drop NaN values (last row will be NaN after shifting)
  df.dropna(inplace=True)

  return df, scalers

def prepare_features_order_book(df):
  # Define columns for scale 
  columns_price = ['mid_price_mean', 'mid_price_min', 'mid_price_max', 'mid_price_last', 'vwap_ask_mean', 'vwap_bid_mean', 'vwap_total_mean']
  columns_price_std = ['mid_price_std']
  columns_spread = ['spread_mean', 'spread_std']
  columns_spread_max = ['spread_max']
  columns_spread_relative = ['relative_spread_mean', 'relative_spread_std']
  columns_volume = ['total_best_ask_volume_mean', 'total_best_ask_volume_std', 'total_best_bid_volume_mean', 'total_best_bid_volume_std']
  columns_volume_max = ['total_best_ask_volume_max', 'total_best_bid_volume_max']
  columns_volume_sum = ['total_best_ask_volume_sum', 'total_best_bid_volume_sum', 'total_best_bid_volume_max']
  columns_volume_cumulative = ['cumulative_delta_volume_last']
  columns_volume_ratio = ['order_book_imbalance_mean', 'order_book_imbalance_std', 'volume_imbalance_ratio_mean', 'volume_imbalance_ratio_std', 'mean_ask_size_mean', 'mean_ask_size_std', 'mean_bid_size_mean', 'mean_bid_size_std', 'std_ask_size_mean', 'std_bid_size_mean']
  columns_market_depth = ['market_depth_ask_mean', 'market_depth_ask_std', 'market_depth_bid_mean', 'market_depth_bid_std']
  columns_liquidity = ['liquidity_pressure_ratio_mean', 'liquidity_pressure_ratio_std']
  columns_volatility = ['realized_volatility']
  
  # Initialize scalers
  scalers = {
    'price': MinMaxScaler(),
    'price_std': StandardScaler(),
    'spread': StandardScaler(),
    'spread_max': RobustScaler(),
    'spread_relative': StandardScaler(),
    'volume': RobustScaler(),# TODO
    'volume_max': RobustScaler(),
    'volume_sum': StandardScaler(),# TODO
    'volume_ratio': StandardScaler(),
    'volume_cumulative': StandardScaler(),
    'market_depth': StandardScaler(),# TODO
    'liquidity': StandardScaler(),
    'volatility': StandardScaler(),# TODO
  }
  
  # Apply different scalers
  df[columns_price] = scalers['price'].fit_transform(df[columns_price])
  df[columns_price_std] = scalers['price_std'].fit_transform(df[columns_price_std])
  df[columns_spread] = scalers['spread'].fit_transform(df[columns_spread])
  df[columns_spread_max] = scalers['spread_max'].fit_transform(df[columns_spread_max])
  df[columns_spread_relative] = scalers['spread_relative'].fit_transform(df[columns_spread_relative])
  df[columns_volume_max] = scalers['volume_max'].fit_transform(df[columns_volume_max])
  df[columns_volume_ratio] = scalers['volume_ratio'].fit_transform(df[columns_volume_ratio])
  df[columns_volume_cumulative] = scalers['volume_cumulative'].fit_transform(df[columns_volume_cumulative])
  df[columns_liquidity] = scalers['liquidity'].fit_transform(df[columns_liquidity])
  
  # Apply log transformation to turnover before scaling
  df[columns_volume] = np.log1p(df[columns_volume])  # log1p to avoid log(0) issues
  df[columns_volume_sum] = np.log1p(df[columns_volume_sum])  # log1p to avoid log(0) issues
  df[columns_market_depth] = np.log1p(df[columns_market_depth])  # log1p to avoid log(0) issues
  df[columns_volatility] = np.log1p(df[columns_volatility])  # log1p to avoid log(0) issues
  
  df[columns_volume] = scalers['volume'].fit_transform(df[columns_volume])
  df[columns_volume_sum] = scalers['volume_sum'].fit_transform(df[columns_volume_sum])
  df[columns_market_depth] = scalers['market_depth'].fit_transform(df[columns_market_depth])
  df[columns_volatility] = scalers['volatility'].fit_transform(df[columns_volatility])
  
  return df, scalers

if __name__ == '__main__':
  symbol = 'BTCUSDT'
  folder_features_kline = f'features/kline/{symbol}/1'
  folder_features_order_book = f'features/order_book/{symbol}'

  df_features_kline = pd.read_csv(f'{folder_features_kline}/features.csv')
  df_features_order_book = pd.read_csv(f'{folder_features_order_book}/features.csv')

  # Prepare features
  df_features_kline_scaled, scalers_features_kline = prepare_features_kline(df_features_kline)
  df_features_order_book_scaled, scalers_features_order_book = prepare_features_order_book(df_features_order_book)

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
  model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(1)
  ])

  # Compile model
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(optimizer=optimizer, loss='mean_squared_error')

  # Train model
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
  model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

  # Predictions
  y_pred = model.predict(X_test)

  # Calculate Errors
  mse = mean_squared_error(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)

  print(f'Mean Squared Error (MSE): {mse:.6f}')
  print(f'Mean Absolute Error (MAE): {mae:.6f}')

  # Convert predictions to a 1D array
  y_pred = y_pred.flatten()

  # Plot actual vs predicted relative price change
  plt.figure(figsize=(12, 6))
  plt.plot(y_test, label="Actual Relative Price Change", color="blue", alpha=0.7)
  plt.plot(y_pred, label="Predicted Relative Price Change", color="red", linestyle="dashed", alpha=0.7)

  plt.xlabel("Time Step")
  plt.ylabel("Relative Price Change")
  plt.title("Actual vs. Predicted Relative Price Change")
  plt.legend()
  plt.show()

  # Save scaled dataset
  df_features_kline_scaled.to_csv(f'{folder_features_kline}/features_scaled.csv')
  df_features_order_book_scaled.to_csv(f'{folder_features_order_book}/features_scaled.csv')
  print('Scaled dataset have been saved.')

  # Save the model and the scaler
  model.save('models/model_kline_lstm.keras')
  print('Model has been saved.')

  # Save the feature and target scalers
  for scaler_name, scaler in scalers_features_kline.items():
    joblib.dump(scaler, f'models/model_kline_scaler_{scaler_name}.pkl')
  for scaler_name, scaler in scalers_features_order_book.items():
    joblib.dump(scaler, f'models/model_order_book_scaler_{scaler_name}.pkl')
  print('Scalers have been saved.')
