import numpy as np
import pandas as pd

# **RSI** (14-period as a common choice)
def calculate_rsi(df, period=14, column='closePrice'):
  delta = df[column].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
  rs = gain / loss
  rs = rs.fillna(0).replace(np.inf, 0)
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

def calc_features_kline_based(df_features, window_short=12, window_long=24):
  # Convert startTime to datetime format (assuming startTime column exists)
  df_features['startTime'] = pd.to_datetime(df_features['startTime'], unit='ms')

  # Ensure numeric columns are properly cast
  df_features['openPrice'] = df_features['openPrice'].astype(float)
  df_features['highPrice'] = df_features['highPrice'].astype(float)
  df_features['lowPrice'] = df_features['lowPrice'].astype(float)
  df_features['closePrice'] = df_features['closePrice'].astype(float)
  df_features['volume'] = df_features['volume'].astype(float)
  df_features['turnover'] = df_features['turnover'].astype(float)

  # Features

  # 1. Price & Return Features

  # 1.1. Price Change (Difference between last close and previous close)
  df_features['priceChange'] = df_features['closePrice'].diff()
  df_features['priceChange_window_short'] = df_features['closePrice'] - df_features['closePrice'].shift(window_short)
  df_features['priceChange_window_long'] = df_features['closePrice'] - df_features['closePrice'].shift(window_long)

  # 1.1.1 Relative price change (percentage change)
  df_features['relativePriceChange'] = df_features["closePrice"].pct_change()
  df_features['relativePriceChange_window_short'] = (df_features['closePrice'] - df_features['closePrice'].shift(window_short)) / df_features['closePrice'].shift(window_short)
  df_features['relativePriceChange_window_long'] = (df_features['closePrice'] - df_features['closePrice'].shift(window_long)) / df_features['closePrice'].shift(window_long)

  # 1.2. Log Return
  df_features['logReturn'] = np.log(df_features['closePrice'] / df_features['closePrice'].shift(1))

  # 1.3. Short-Term Moving Averages
  df_features['SMA_short'] = df_features['closePrice'].rolling(window=window_short).mean()
  df_features['SMA_long'] = df_features['closePrice'].rolling(window=window_long).mean()

  # 1.4. Short-Term Exponential Moving Averages
  df_features['EMA_short'] = df_features['closePrice'].ewm(span=window_short, adjust=False).mean()
  df_features['EMA_long'] = df_features['closePrice'].ewm(span=window_long, adjust=False).mean()

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

  # 3.2. Standard Deviation of Returns
  df_features['stdReturn_short'] = df_features['logReturn'].rolling(window=window_short).std()
  df_features['stdReturn_long'] = df_features['logReturn'].rolling(window=window_long).std()

  # 4. Momentum Indicators

  # **RSI**
  df_features['RSI_short'] = calculate_rsi(df_features, period=window_short)
  df_features['RSI_long'] = calculate_rsi(df_features, period=window_long)

  # **MACD**
  signal_period = max(round(window_long / 3), 3)
  df_features['MACD_line'], df_features['MACD_signal'], df_features['MACD_histogram'] = calculate_macd(df_features, fast_period=window_short, slow_period=window_long, signal_period=signal_period)

  # **Stochastic Oscillator** (%K and %D)
  df_features['Stochastic_K_short'], df_features['Stochastic_D_short'] = calculate_stochastic(df_features, period=window_short)
  df_features['Stochastic_K_long'], df_features['Stochastic_D_long'] = calculate_stochastic(df_features, period=window_long)

  # Calculate ROC
  df_features['ROC_short'] = calculate_roc(df_features, period=window_short)
  df_features['ROC_long'] = calculate_roc(df_features, period=window_long)

  # Convert startTime to timestamp
  df_features['startTime'] = df_features['startTime'].astype('int64') // 10**6
  
  # Order by timestamp ascending
  df_features = df_features.sort_values(by='startTime')

  return df_features
