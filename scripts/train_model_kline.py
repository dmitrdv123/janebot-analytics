import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

def prepare_features_kline(df):
  # Define model output  
  df['futureRelativePriceChange'] = df['relativePriceChange'].shift(-1)
  df['futurePriceChange'] = df['priceChange'].shift(-1)
  df['futureClosePrice'] = df['closePrice'].shift(-1)

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
  df[target_columns] = scalers['target'].fit_transform(df[target_columns])
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

  # Drop NaN values (last row will be NaN after shifting)
  df.dropna(inplace=True)

  return df, scalers

if __name__ == '__main__':
  symbol = 'BTCUSDT'
  folder_features_kline = f'features/kline/{symbol}/1'

  df_features_kline = pd.read_csv(f'{folder_features_kline}/features.csv')

  # Order by timestamp ascending
  df_features_kline = df_features_kline.sort_values(by='startTime')

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
  model.save('models/model_kline_lstm.keras')
  print('Model has been saved.')

  # Save the feature and target scalers
  for scaler_name, scaler in scalers_features_kline.items():
    joblib.dump(scaler, f'models/model_kline_scaler_{scaler_name}.pkl')
  print('Scalers have been saved.')
