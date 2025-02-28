import numpy as np
import tensorflow as tf
import joblib

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from utils_features import calc_features_kline_based
from utils import load_data

def load_model_and_scalers(path='models', model='model_kline_lstm.keras'):
  # Load the trained LSTM model
  model = tf.keras.models.load_model(f'{path}/{model}')
  
  # Load the scalers
  scalers = {
    'target': joblib.load(f'{path}/model_kline_scaler_target.pkl'),
    'price': joblib.load(f'{path}/model_kline_scaler_price.pkl'),
    'price_change': joblib.load(f'{path}/model_kline_scaler_price_change.pkl'),
    'volume': joblib.load(f'{path}/model_kline_scaler_volume.pkl'),
    'turnover': joblib.load(f'{path}/model_kline_scaler_turnover.pkl'),
    'returns': joblib.load(f'{path}/model_kline_scaler_returns.pkl'),
    'range': joblib.load(f'{path}/model_kline_scaler_range.pkl')
  }
  return model, scalers

def load_and_prepare_data(symbol='BTCUSDT', interval=1):
  # Load the Kline Data
  df_data = load_data(f'data/kline/{symbol}/{interval}')
  
  # Ensure numeric columns are properly cast
  df_data['startTime'] = df_data['startTime'].astype(float)
  df_data['openPrice'] = df_data['openPrice'].astype(float)
  df_data['highPrice'] = df_data['highPrice'].astype(float)
  df_data['lowPrice'] = df_data['lowPrice'].astype(float)
  df_data['closePrice'] = df_data['closePrice'].astype(float)
  df_data['volume'] = df_data['volume'].astype(float)
  df_data['turnover'] = df_data['turnover'].astype(float)
  
  # Sort by timestamp
  df_data = df_data.sort_values(by='startTime')
  
  return df_data

def prepare_features_kline(df, scalers):
  # Define model output  
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

  # Apply different scalers
  df[target_columns] = scalers['target'].transform(df[target_columns])
  df[price_columns] = scalers['price'].transform(df[price_columns])
  df[price_change_column] = scalers['price_change'].transform(df[price_change_column])
  df[volume_columns] = scalers['volume'].transform(df[volume_columns])
  df[returns_columns] = scalers['returns'].transform(df[returns_columns])
  df[range_columns] = scalers['range'].transform(df[range_columns])

  # Apply log transformation to turnover before scaling
  df[turnover_column] = np.log1p(df[turnover_column])  # log1p to avoid log(0) issues
  df[turnover_column] = scalers['turnover'].transform(df[turnover_column])

  # Apply scale to categorical features
  df['hourOfDay'] = df['hourOfDay'] / 23  # Normalize to [0,1]
  df['dayOfWeek'] = df['dayOfWeek'] / 6    # Normalize to [0,1] (0=Monday, 6=Sunday)
  df['weekOfYear'] = df['weekOfYear'] / 51 # Normalize to [0,1]
  df['monthOfYear'] = df['monthOfYear'] / 11 # Normalize to [0,1] (0=Jan, 11=Dec)
  df['minuteOfHour'] = df['minuteOfHour'] / 59 # Normalize to [0,1]

  # Drop NaN values (last row will be NaN after shifting)
  df.dropna(inplace=True)

  return df

if __name__ == '__main__':
  # Load model and scalers
  model, scalers = load_model_and_scalers()

  # Load data
  df_data = load_and_prepare_data(symbol='BTCUSDT', interval=5)
  
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

  # Lists to store results
  actual_values = []
  predicted_values = []
  
  window_size = 16
  step_size = 1

  # Sliding window backtest
  for start_idx in range(0, len(df_data) - window_size, step_size):
    end_idx = start_idx + window_size + 1  # +1 to include future value
    
    # Extract window
    df_window = df_data.iloc[start_idx:end_idx].copy()
    
    # Calculate features
    df_features = calc_features_kline_based(df_window)
    
    df_features_scaled = prepare_features_kline(df_features, scalers)
    
    if len(df_features) < 1:  # Skip if window is too small after dropping NaNs
      continue
    
    # Prepare input data
    X = df_features[input_columns].values
    y = df_features[target_column].values
    
    # Convert to proper float type
    X = np.array(X, dtype=np.float32)  # Ensure it's a NumPy array with float32
    y = np.array(y, dtype=np.float32)  # Ensure target is also float32
    
    # Reshape data for LSTM (samples, time steps, features)
    X = X.reshape(X.shape[0], 1, X.shape[1])
      
    # Predict
    y_pred = model.predict(X).flatten()
    
    predicted_values.extend(y_pred)
    actual_values.extend(y)
    
    # Calculate and print errors in each iteration
    mse = mean_squared_error([actual_values[-1]], [predicted_values[-1]])
    mae = mean_absolute_error([actual_values[-1]], [predicted_values[-1]])
    print(f"Iteration : MSE={mse:.6f}, MAE={mae:.6f}")

    # Plot actual vs. predicted price dynamically
    plt.clf()
    plt.plot(actual_values, label="Actual Close Price", color="blue", alpha=0.7)
    plt.plot(predicted_values, label="Predicted Close Price", color="red", linestyle="dashed", alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Close Price")
    plt.title("Actual vs. Predicted Close Price Over Time")
    plt.legend()
    plt.pause(0.01)  # Pause to update plot dynamically

  plt.show()