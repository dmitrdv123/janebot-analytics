import os
import numpy as np
import pandas as pd

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

# **RSI** (14-period as a common choice)
def calculate_rsi(df, period=14):
    delta = df['closePrice'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
  
# **MACD** (12-period and 26-period EMAs, and 9-period Signal line)
def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    macd_line = df['closePrice'].ewm(span=fast_period, adjust=False).mean() - df['closePrice'].ewm(span=slow_period, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# **Stochastic Oscillator** (14-period %K and %D)
def calculate_stochastic(df, period=14):
    low_min = df['lowPrice'].rolling(window=period).min()
    high_max = df['highPrice'].rolling(window=period).max()
    k_line = 100 * (df['closePrice'] - low_min) / (high_max - low_min)
    d_line = k_line.rolling(window=3).mean()  # 3-period %D
    return k_line, d_line

# **Rate of Change (ROC)** - Percentage change over n-periods
def calculate_roc(df, period=14):
    roc = ((df['closePrice'] - df['closePrice'].shift(period)) / df['closePrice'].shift(period)) * 100
    return roc

# Load the Kline Data
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

# Features

# 1. Price & Return Features

# 1.1. Price Change (Difference between last close and previous close)
kline_df['priceChange'] = kline_df['closePrice'].diff()

# 1.2. Log Return
kline_df['logReturn'] = np.log(kline_df['closePrice'] / kline_df['closePrice'].shift(1))

# 1.3. Short-Term Moving Averages (5 and 10-period)
kline_df['SMA_5'] = kline_df['closePrice'].rolling(window=5).mean()
kline_df['SMA_10'] = kline_df['closePrice'].rolling(window=10).mean()

# 1.4. Short-Term Exponential Moving Averages (5 and 10-period)
kline_df['EMA_5'] = kline_df['closePrice'].ewm(span=5, adjust=False).mean()
kline_df['EMA_10'] = kline_df['closePrice'].ewm(span=10, adjust=False).mean()

# 2. Time-Based Features

# 2.1. Extract Hour of the Day (0-23)
kline_df['hourOfDay'] = kline_df['startTime'].dt.hour

# 2.2. Extract Day of the Week (0-6, where 0 is Monday)
kline_df['dayOfWeek'] = kline_df['startTime'].dt.dayofweek

# 2.3. Extract Week of the Year (1-52)
kline_df['weekOfYear'] = kline_df['startTime'].dt.isocalendar().week

# 2.4. Extract Month of the Year (1-12)
kline_df['monthOfYear'] = kline_df['startTime'].dt.month

# 2.5. Extract Minute of the Hour (0-59)
kline_df['minuteOfHour'] = kline_df['startTime'].dt.minute

# 2.6. Identify if the day is a weekend (0 = False, 1 = True)
kline_df['isWeekend'] = kline_df['dayOfWeek'].isin([5, 6]).astype(int)

# 3. Volatility Features

# 3.1. High-Low Range (Price Volatility in a Minute)
kline_df['highLowRange'] = kline_df['highPrice'] - kline_df['lowPrice']

# 3.2. Standard Deviation of Returns (Rolling window of 5,10 minutes)
kline_df['stdReturn_5m'] = kline_df['logReturn'].rolling(window=5).std()
kline_df['stdReturn_10m'] = kline_df['logReturn'].rolling(window=10).std()

# 4. Momentum Indicators

# **RSI** (14-period as a common choice)
kline_df['RSI_14'] = calculate_rsi(kline_df, period=14)

# **MACD** (12-period and 26-period EMAs, and 9-period Signal line)
kline_df['MACD_line'], kline_df['MACD_signal'], kline_df['MACD_histogram'] = calculate_macd(kline_df)

# **Stochastic Oscillator** (14-period %K and %D)
kline_df['Stochastic_K'], kline_df['Stochastic_D'] = calculate_stochastic(kline_df)

# Calculate ROC with 14-period window
kline_df['ROC_14'] = calculate_roc(kline_df, period=14)

# Drop NaN values from rolling calculations
kline_df.dropna(inplace=True)

# Print sample of feature dataframe
print(kline_df.head())

# Save the features to a CSV file
kline_df.to_csv('features/kline/1m/features.csv', index=False)
