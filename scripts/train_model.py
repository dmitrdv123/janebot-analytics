import os
import joblib
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load all the Kline data files from the 'data/kline' directory
def load_kline_data(directory='data/kline/1m'):
  kline_df = pd.DataFrame()

  # Loop through all CSV files in the directory
  for filename in os.listdir(directory):
    if filename.endswith('.csv'):
      print(f'Loading data from {filename}')

      file_path = os.path.join(directory, filename)
      # Load the file into a DataFrame and append it
      temp_df = pd.read_csv(file_path)

      # Optionally, you can extract the symbol and interval from the filename
      # This depends on your file naming convention, e.g., extracting `symbol` and `interval`
      # Example: file_name: 'BTCUSD_1h_1634582400000_1634586000000.csv'
      symbol, interval, timestamp_begin, timestamp_end = filename.replace('.csv', '').split('_')
      temp_df['symbol'] = symbol
      temp_df['interval'] = interval
      temp_df['timestamp_begin'] = timestamp_begin
      temp_df['timestamp_end'] = timestamp_end

      # Concatenate the DataFrame with the existing data
      kline_df = pd.concat([kline_df, temp_df], ignore_index=True)

  return kline_df

# Step: Load the Kline Data
kline_df = load_kline_data('data/kline/1m')

# Convert startTime to datetime format (assuming startTime column exists)
kline_df['startTime'] = pd.to_datetime(kline_df['startTime'], unit='ms')

# Ensure numeric columns are properly cast
kline_df['openPrice'] = kline_df['openPrice'].astype(float)
kline_df['highPrice'] = kline_df['highPrice'].astype(float)
kline_df['lowPrice'] = kline_df['lowPrice'].astype(float)
kline_df['closePrice'] = kline_df['closePrice'].astype(float)
kline_df['volume'] = kline_df['volume'].astype(float)
kline_df['turnover'] = kline_df['turnover'].astype(float)

# Step: Feature Engineering
# Calculate price change (target variable)
kline_df['price_change'] = kline_df['closePrice'].shift(-1) - kline_df['closePrice']

# Calculate price range
kline_df['price_range'] = kline_df['highPrice'] - kline_df['lowPrice']

# Calculate moving averages (SMA)
kline_df['SMA_5'] = kline_df['closePrice'].rolling(window=5).mean()
kline_df['SMA_15'] = kline_df['closePrice'].rolling(window=15).mean()

# Calculate exponential moving averages (EMA)
kline_df['EMA_5'] = kline_df['closePrice'].ewm(span=5, adjust=False).mean()
kline_df['EMA_15'] = kline_df['closePrice'].ewm(span=15, adjust=False).mean()

# Calculate volume-based features
kline_df['volume_change'] = kline_df['volume'].pct_change()  # Percentage change in volume
kline_df['turnover_change'] = kline_df['turnover'].pct_change()  # Percentage change in turnover

# Drop missing values
kline_df = kline_df.dropna()

# Step: Normalizing Price Data Using MinMaxScaler
scaler_price = MinMaxScaler()
scaler_volume = MinMaxScaler()
scaler_indicator = MinMaxScaler()
scaler_target = MinMaxScaler()

# Normalize price-related columns
kline_df[['openPrice', 'highPrice', 'lowPrice', 'closePrice']] = scaler_price.fit_transform(
  kline_df[['openPrice', 'highPrice', 'lowPrice', 'closePrice']]
)

# Normalize other columns as well if needed
kline_df[['volume', 'turnover', 'volume_change', 'turnover_change']] = scaler_volume.fit_transform(
  kline_df[['volume', 'turnover', 'volume_change', 'turnover_change']]
)

kline_df[['SMA_5', 'SMA_15', 'EMA_5', 'EMA_15']] = scaler_indicator.fit_transform(
  kline_df[['SMA_5', 'SMA_15', 'EMA_5', 'EMA_15']]
)

# Normalize target variable 'price_change'
y_actual = kline_df['price_change']
y_actual_normalized = scaler_target.fit_transform(y_actual.values.reshape(-1, 1))

# Step: Prepare Features and Target Variable
# Select features and target
X = kline_df[['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover', 'SMA_5', 'SMA_15', 'EMA_5', 'EMA_15', 'volume_change', 'turnover_change']]
y = kline_df['price_change']  # Target: Predict price change

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_actual_normalized, test_size=0.2, shuffle=False)

# Step: Build and Train TensorFlow Model
model = tf.keras.Sequential([
  tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(1)  # Single output (predicted price change)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step: Evaluate the Model
# Make predictions
y_pred = model.predict(X_test)

# Reverse normalization for predictions and actual values
y_pred_actual = scaler_target.inverse_transform(y_pred)
y_test_actual = scaler_target.inverse_transform(y_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test_actual, y_pred_actual)
print(f'Mean Squared Error: {mse}')

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test_actual, y_pred_actual)
print(f'Mean Absolute Error: {mae}')

# Step: Save the trained model
model_save_path = os.path.join('models', 'kline_price_model.keras')
model.save(model_save_path)
print(f'Model saved to: {model_save_path}')

# Save scalers
joblib.dump(scaler_price, 'models/scaler_price.pkl')
joblib.dump(scaler_volume, 'models/scaler_volume.pkl')
joblib.dump(scaler_indicator, 'models/scaler_indicator.pkl')
joblib.dump(scaler_target, 'models/scaler_target.pkl')
print(f'All scaler saved')
