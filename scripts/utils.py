import os

# Create a function to ensure directories exist
def ensure_directory(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

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
