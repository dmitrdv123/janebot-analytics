import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
  columns_volume_sum = ['total_best_ask_volume_sum', 'total_best_bid_volume_sum']
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

def merge_feature(df_features_kline_scaled, df_features_order_book_scaled):
    # Add prefix
  df_features_kline_scaled = df_features_kline_scaled.add_prefix('kline_')
  df_features_order_book_scaled = df_features_order_book_scaled.add_prefix('order_book_')
  
  # Join features
  df_features_kline_scaled['kline_startTime'] = pd.to_datetime(df_features_kline_scaled['kline_startTime'], unit='ms')
  df_features_order_book_scaled['order_book_timestamp'] = pd.to_datetime(df_features_order_book_scaled['order_book_timestamp'], unit='ms')
   
  df_merged = pd.merge_asof(
    df_features_kline_scaled.sort_values('kline_startTime'),
    df_features_order_book_scaled.sort_values('order_book_timestamp'),
    left_on='kline_startTime',
    right_on='order_book_timestamp',
    direction='backward'
  )

  min_timestamp = df_features_order_book_scaled['order_book_timestamp'].min()
  max_timestamp = df_features_order_book_scaled['order_book_timestamp'].max()
  df_merged = df_merged[(df_merged['kline_startTime'] >= min_timestamp) & (df_merged['kline_startTime'] <= max_timestamp)]

  # Convert startTime to timestamp
  df_merged['kline_startTime'] = df_merged['kline_startTime'].astype('int64') // 10**6
  df_merged['order_book_timestamp'] = df_merged['order_book_timestamp'].astype('int64') // 10**6
  
  return df_merged

if __name__ == '__main__':
  # Load datasets
  symbol = 'BTCUSDT'
  folder_features_kline = f'features/kline/{symbol}/1'
  folder_features_order_book = f'features/order_book/{symbol}'

  df_features_kline = pd.read_csv(f'{folder_features_kline}/features.csv')
  df_features_order_book = pd.read_csv(f'{folder_features_order_book}/features.csv')

  # Prepare kline features
  df_features_kline_scaled, scalers_features_kline = prepare_features_kline(df_features_kline)

  # Prepare order book features
  df_features_order_book_scaled, scalers_features_order_book = prepare_features_order_book(df_features_order_book)
  
  # Merge features
  df_features_merged = merge_feature(df_features_kline, df_features_order_book)
  df_features_merged.to_csv(f'features/merged/{symbol}/features.csv')

  # Define feature columns
  columns_input_kline = [
    'kline_openPrice', 
    'kline_highPrice', 
    'kline_lowPrice', 
    'kline_closePrice', 
    'kline_volume', 
    'kline_turnover', 
    'kline_priceChange', 
    'kline_relativePriceChange', 
    'kline_logReturn', 
    'kline_SMA_5', 
    'kline_SMA_10', 
    'kline_EMA_5', 
    'kline_EMA_10',
    'kline_hourOfDay', 
    'kline_dayOfWeek', 
    'kline_weekOfYear', 
    'kline_monthOfYear', 
    'kline_minuteOfHour', 
    'kline_isWeekend', 
    'kline_highLowRange', 
    'kline_stdReturn_5m', 
    'kline_stdReturn_10m', 
    'kline_RSI_14', 
    'kline_MACD_line', 
    'kline_MACD_signal', 
    'kline_MACD_histogram', 
    'kline_Stochastic_K', 
    'kline_Stochastic_D', 
    'kline_ROC_14',
  ]
  
  columns_input_order_book = [
    'order_book_mid_price_mean',
    'order_book_mid_price_std',
    'order_book_mid_price_min',
    'order_book_mid_price_max',
    'order_book_mid_price_last',
    'order_book_spread_mean',
    'order_book_spread_std',
    'order_book_spread_max',
    'order_book_relative_spread_mean',
    'order_book_relative_spread_std',
    'order_book_total_best_ask_volume_mean',
    'order_book_total_best_ask_volume_std',
    'order_book_total_best_ask_volume_sum',
    'order_book_total_best_ask_volume_max',
    'order_book_total_best_bid_volume_mean',
    'order_book_total_best_bid_volume_std',
    'order_book_total_best_bid_volume_sum',
    'order_book_total_best_bid_volume_max',
    'order_book_market_depth_ask_mean',
    'order_book_market_depth_ask_std',
    'order_book_market_depth_bid_mean',
    'order_book_market_depth_bid_std',
    'order_book_order_book_imbalance_mean',
    'order_book_order_book_imbalance_std',
    'order_book_vwap_ask_mean',
    'order_book_vwap_bid_mean',
    'order_book_vwap_total_mean',
    'order_book_volume_imbalance_ratio_mean',
    'order_book_volume_imbalance_ratio_std',
    'order_book_cumulative_delta_volume_last',
    'order_book_liquidity_pressure_ratio_mean',
    'order_book_liquidity_pressure_ratio_std',
    'order_book_mean_ask_size_mean',
    'order_book_mean_ask_size_std',
    'order_book_mean_bid_size_mean',
    'order_book_mean_bid_size_std',
    'order_book_std_ask_size_mean',
    'order_book_std_bid_size_mean',
    'order_book_realized_volatility',
  ]
  
  # Target variable (future relative price change)
  target_column = 'kline_futureClosePrice'
  
  # Prepare input and output data
  X_kline = df_features_merged[columns_input_kline].values
  X_orderbook = df_features_merged[columns_input_order_book].values
  y = df_features_merged[target_column].values
  
  # Train-Test Split
  X_kline_train, X_kline_test, X_orderbook_train, X_orderbook_test, y_train, y_test = train_test_split(X_kline, X_orderbook, y, test_size=0.2, shuffle=False)

  # Reshape for LSTM
  X_kline_train = X_kline_train.reshape(X_kline_train.shape[0], 1, X_kline_train.shape[1])
  X_kline_test = X_kline_test.reshape(X_kline_test.shape[0], 1, X_kline_test.shape[1])

  # Reshape for CNN
  X_orderbook_train = X_orderbook_train.reshape(X_orderbook_train.shape[0], X_orderbook_train.shape[1], 1)
  X_orderbook_test = X_orderbook_test.reshape(X_orderbook_test.shape[0], X_orderbook_test.shape[1], 1)

  # Define LSTM model for Kline
  input_kline = tf.keras.layers.Input(shape=(1, X_kline_train.shape[2]))
  layer_kline = tf.keras.layers.LSTM(256, activation='tanh', return_sequences=True)(input_kline)  # More units + return_sequences=True
  layer_kline = tf.keras.layers.LSTM(128, activation='tanh', return_sequences=False)(layer_kline)  # Additional LSTM layer

  # Define CNN model for Order Book with more filters
  input_orderbook = tf.keras.layers.Input(shape=(X_orderbook_train.shape[1], 1))
  layer_order_book = tf.keras.layers.Conv1D(256, kernel_size=3, activation='relu')(input_orderbook)
  layer_order_book = tf.keras.layers.MaxPooling1D(pool_size=2)(layer_order_book)
  layer_order_book = tf.keras.layers.Flatten()(layer_order_book)
  layer_order_book = tf.keras.layers.Dense(128, activation='relu')(layer_order_book)

  # Merge LSTM and CNN
  merged = tf.keras.layers.concatenate([layer_kline, layer_order_book])
  merged = tf.keras.layers.Dense(128, activation='relu')(merged)  # Added Dense layer with more units
  output = tf.keras.layers.Dense(1)(merged)

  # Define model
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
  model = tf.keras.Model(inputs=[input_kline, input_orderbook], outputs=output)
  model.compile(optimizer=optimizer, loss='mean_squared_error')

  # Train model
  lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
  model.fit(
    [X_kline_train, X_orderbook_train], 
    y_train, 
    epochs=1000, 
    batch_size=32, 
    validation_data=([X_kline_test, X_orderbook_test], y_test), 
    verbose=1, 
    callbacks=[early_stopping, lr_scheduler]
  )

  # Predictions
  y_pred = model.predict([X_kline_test, X_orderbook_test])

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
  print('Scaled dataset have been saved.')

  # Save the model and the scaler
  model.save('models/model_fusion.keras')
  print('Model has been saved.')

  # Save the feature and target scalers
  for scaler_name, scaler in scalers_features_kline.items():
    joblib.dump(scaler, f'models/model_fusion_kline_scaler_{scaler_name}.pkl')
  for scaler_name, scaler in scalers_features_order_book.items():
    joblib.dump(scaler, f'models/model_fusion_order_book_scaler_{scaler_name}.pkl')
  print('Scalers have been saved.')
