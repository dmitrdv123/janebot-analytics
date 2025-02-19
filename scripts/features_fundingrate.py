import os
import pandas as pd

# Load all the Kline data files from the 'data/fundingrate' directory
def load_data(directory='data/fundingrate'):
  df_features = pd.DataFrame()

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
      symbol, timestamp_begin, timestamp_end = filename.replace('.csv', '').split('_')
      temp_df['symbol'] = symbol
      temp_df['timestamp_begin'] = timestamp_begin
      temp_df['timestamp_end'] = timestamp_end

      # Concatenate the DataFrame with the existing data
      df_features = pd.concat([df_features, temp_df], ignore_index=True)

  return df_features

# Load the Kline Data
df_features = load_data('data/fundingrate')

# Convert fundingRateTimestamp to datetime format (assuming fundingRateTimestamp column exists)
df_features['fundingRateTimestamp'] = pd.to_datetime(df_features['fundingRateTimestamp'], unit='ms')

# Ensure numeric columns are properly cast
df_features['fundingRate'] = df_features['fundingRate'].astype(float)

# Features

# Calculate rolling mean and volatility for different windows
df_features['fundingRateMean_5m'] = df_features['fundingRate'].rolling(window=5).mean()
df_features['fundingRateMean_10m'] = df_features['fundingRate'].rolling(window=10).mean()
df_features['fundingRateVolatility_5m'] = df_features['fundingRate'].rolling(window=5).std()
df_features['fundingRateVolatility_10m'] = df_features['fundingRate'].rolling(window=10).std()

# Cumulative funding rate
df_features['cumulativeFundingRate'] = df_features['fundingRate'].cumsum()

# First derivative (change)
df_features['fundingRateChange'] = df_features['fundingRate'].diff()

# Rate of change (percentage change)
df_features['fundingRateROC'] = df_features['fundingRate'].pct_change()

# Sentiment based on funding rate direction
df_features['sentiment'] = df_features['fundingRate'].apply(lambda x: 'long-biased' if x > 0 else 'short-biased')

# Lagged feature
df_features['fundingRateLag1'] = df_features['fundingRate'].shift(1)

# Print sample of feature dataframe
print(df_features.head())

# Save the features to a CSV file
df_features.to_csv('features/fundingrate/features.csv', index=False)
