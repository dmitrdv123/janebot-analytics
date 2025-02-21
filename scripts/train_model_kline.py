from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import tensorflow as tf

if __name__ == '__main__':
  symbol = 'BTCUSDT'

  df = pd.read_csv(f'features/kline/{symbol}/1/features.csv')
   
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

  # Define model output  
  df['futureRelativePriceChange'] = df['relativePriceChange'].shift(-1)
  df['futurePriceChange'] = df['priceChange'].shift(-1)
  df['futureClosePrice'] = df['closePrice'].shift(-1)
  
  # Drop NaN values (last row will be NaN after shifting)
  df.dropna(inplace=True)
  
  # Define feature columns
  feature_columns = [
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
  X = df[feature_columns].values  # Features
  y = df[target_column].values    # Target (future relative price change)
  
  # Train-Test Split (80% training, 20% testing)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

  # Define TensorFlow model
  model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
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
