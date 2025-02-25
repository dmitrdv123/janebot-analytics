import numpy as np
import pandas as pd

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

def calc_features_kline_based(df_features):
  # Convert startTime to datetime format (assuming startTime column exists)
  df_features['startTime'] = pd.to_datetime(df_features['startTime'], unit='ms')

  # Ensure numeric columns are properly cast
  df_features['openPrice'] = df_features['openPrice'].astype(float)
  df_features['highPrice'] = df_features['highPrice'].astype(float)
  df_features['lowPrice'] = df_features['lowPrice'].astype(float)
  df_features['closePrice'] = df_features['closePrice'].astype(float)

  # Features

  # 1. Price & Return Features

  # 1.1. Price Change (Difference between last close and previous close)
  df_features['priceChange'] = df_features['closePrice'].diff()

  # 1.1.1 Relative price change (percentage change)
  df_features["relativePriceChange"] = df_features["closePrice"].pct_change()

  # 1.2. Log Return
  df_features['logReturn'] = np.log(df_features['closePrice'] / df_features['closePrice'].shift(1))

  # 1.3. Short-Term Moving Averages (5 and 10-period)
  df_features['SMA_5'] = df_features['closePrice'].rolling(window=5).mean()
  df_features['SMA_10'] = df_features['closePrice'].rolling(window=10).mean()

  # 1.4. Short-Term Exponential Moving Averages (5 and 10-period)
  df_features['EMA_5'] = df_features['closePrice'].ewm(span=5, adjust=False).mean()
  df_features['EMA_10'] = df_features['closePrice'].ewm(span=10, adjust=False).mean()

  # 2. Time-Based Features

  # 2.1. Extract Hour of the Day (0-23)
  df_features['hourOfDay'] = df_features['startTime'].dt.hour

  # 2.2. Extract Day of the Week (0-6, where 0 is Monday)
  df_features['dayOfWeek'] = df_features['startTime'].dt.dayofweek

  # 2.3. Extract Week of the Year (1-52)
  df_features['weekOfYear'] = df_features['startTime'].dt.isocalendar().week

  # 2.4. Extract Month of the Year (1-12)
  df_features['monthOfYear'] = df_features['startTime'].dt.month

  # 2.5. Extract Minute of the Hour (0-59)
  df_features['minuteOfHour'] = df_features['startTime'].dt.minute

  # 2.6. Identify if the day is a weekend (0 = False, 1 = True)
  df_features['isWeekend'] = df_features['dayOfWeek'].isin([5, 6]).astype(int)

  # 3. Volatility Features

  # 3.1. High-Low Range (Price Volatility in a Minute)
  df_features['highLowRange'] = df_features['highPrice'] - df_features['lowPrice']

  # 3.2. Standard Deviation of Returns (Rolling window of 5,10 minutes)
  df_features['stdReturn_5m'] = df_features['logReturn'].rolling(window=5).std()
  df_features['stdReturn_10m'] = df_features['logReturn'].rolling(window=10).std()

  # 4. Momentum Indicators

  # **RSI** (14-period as a common choice)
  df_features['RSI_14'] = calculate_rsi(df_features, period=14)

  # **MACD** (12-period and 26-period EMAs, and 9-period Signal line)
  df_features['MACD_line'], df_features['MACD_signal'], df_features['MACD_histogram'] = calculate_macd(df_features)

  # **Stochastic Oscillator** (14-period %K and %D)
  df_features['Stochastic_K'], df_features['Stochastic_D'] = calculate_stochastic(df_features)

  # Calculate ROC with 14-period window
  df_features['ROC_14'] = calculate_roc(df_features, period=14)

  # Convert startTime to timestamp
  df_features['startTime'] = df_features['startTime'].astype('int64') // 10**6
  
  # Order by timestamp ascending
  df_features = df_features.sort_values(by='startTime')

  return df_features
