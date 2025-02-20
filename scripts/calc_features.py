import json
import os
import numpy as np
import pandas as pd

from utils import ensure_directory

def load_data(base_folder_path: str, file_extension: str = ".csv"):
  """
  Downloads all files from subfolders (organized by date) inside a given folder and combines them into a single dataset (DataFrame).

  Args:
  - base_folder_path (str): Path to the base folder containing subfolders (named by date).
  - file_extension (str): Extension of files to download (default is ".csv").

  Returns:
  - pd.DataFrame: Combined dataset of all files in the subfolders.
  """
  all_files = []

  # Check if base folder exists
  if not os.path.exists(base_folder_path):
    raise FileNotFoundError(f"The base folder {base_folder_path} does not exist.")

  # Iterate over the subfolders (which represent dates)
  for subfolder_name in os.listdir(base_folder_path):
    subfolder_path = os.path.join(base_folder_path, subfolder_name)

    # Check if the subfolder is indeed a directory (i.e., not a file)
    if os.path.isdir(subfolder_path):
      print(f"Reading from subfolder: {subfolder_path}")

      # Iterate over the files in the subfolder
      for filename in os.listdir(subfolder_path):
        # Filter based on file extension (e.g., CSV)
        if filename.endswith(file_extension):
          file_path = os.path.join(subfolder_path, filename)
          print(f"Reading file: {file_path}")
          # Read file (assuming CSV for simplicity)
          df = pd.read_csv(file_path)
          all_files.append(df)

  # Combine all files into a single DataFrame
  if all_files:
    combined_data = pd.concat(all_files, ignore_index=True)
    return combined_data
  else:
    raise ValueError("No files found in the subfolders with the specified extension.")

def save_features(df, base_folder, symbol, interval=None):
  if interval:
    date_folder = f'{base_folder}/{symbol}/{interval}'
  else:
    date_folder = f'{base_folder}/{symbol}'
  ensure_directory(date_folder)
  filename = f'{date_folder}/features.csv'
  df.to_csv(filename, index=False)
  print(f'Data saved to {filename}')

def calc_features_kline(symbol, interval):
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
  df_features = load_data(f'data/kline/{symbol}/{interval}')

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

  # Save the features to a CSV file
  save_features(df_features, 'features/kline', symbol, interval)

def calc_features_order_book(symbol):
  def process_order_book(file_path, top_n_levels=20):
      '''
      Processes a large order book file line by line and extracts relevant features including VWAP and volume-based metrics.

      Args:
      - file_path (str): Path to the order book text file.
      - top_n_levels (int): Number of top levels to consider for depth and imbalance.

      Returns:
      - pd.DataFrame: DataFrame with extracted features.
      '''

      features = []
      cumulative_delta_volume = 0  # Initialize CDV

      with open(file_path, 'r') as file:
          for line in file:
              try:
                  order_book = json.loads(line.strip())  # Load JSON from each line
                  ts = order_book['ts']  # Timestamp
                  asks = order_book['data']['a']  # Ask side
                  bids = order_book['data']['b']  # Bid side

                  # Convert string prices and quantities to floats
                  asks = [(float(price), float(qty)) for price, qty in asks]
                  bids = [(float(price), float(qty)) for price, qty in bids]

                  # Ensure there are valid bids and asks
                  if not asks or not bids:
                      continue

                  # Best Ask & Best Bid
                  best_ask_price, best_ask_qty = asks[0]
                  best_bid_price, best_bid_qty = bids[0]

                  # Mid-price
                  mid_price = (best_ask_price + best_bid_price) / 2

                  # Spread and Relative Spread
                  spread = best_ask_price - best_bid_price
                  relative_spread = spread / mid_price if mid_price != 0 else 0

                  # Total volume at best bid & ask
                  total_best_ask_volume = sum(qty for _, qty in asks[:top_n_levels])
                  total_best_bid_volume = sum(qty for _, qty in bids[:top_n_levels])

                  # Market Depth (sum of top N levels' volume)
                  market_depth_ask = sum(qty for _, qty in asks[:top_n_levels])
                  market_depth_bid = sum(qty for _, qty in bids[:top_n_levels])

                  # Order Book Imbalance (OBI)
                  total_volume = market_depth_ask + market_depth_bid
                  order_book_imbalance = (market_depth_bid - market_depth_ask) / total_volume if total_volume != 0 else 0

                  # VWAP Calculation (Using Top N levels)
                  vwap_ask = sum(price * qty for price, qty in asks[:top_n_levels]) / total_best_ask_volume if total_best_ask_volume > 0 else best_ask_price
                  vwap_bid = sum(price * qty for price, qty in bids[:top_n_levels]) / total_best_bid_volume if total_best_bid_volume > 0 else best_bid_price
                  vwap_total = (vwap_ask + vwap_bid) / 2  # Overall VWAP

                  # Volume Imbalance Ratio (VIR)
                  volume_imbalance_ratio = (total_best_bid_volume - total_best_ask_volume) / (total_best_bid_volume + total_best_ask_volume) if (total_best_bid_volume + total_best_ask_volume) != 0 else 0

                  # Cumulative Delta Volume (CDV) - Running sum of (Bid Volume - Ask Volume)
                  delta_volume = total_best_bid_volume - total_best_ask_volume
                  cumulative_delta_volume += delta_volume

                  # Liquidity Pressure Ratio (LPR)
                  liquidity_pressure_ratio = total_best_bid_volume / total_best_ask_volume if total_best_ask_volume > 0 else np.inf

                  # Mean & Std of Order Sizes
                  ask_sizes = [qty for _, qty in asks[:top_n_levels]]
                  bid_sizes = [qty for _, qty in bids[:top_n_levels]]

                  mean_ask_size = np.mean(ask_sizes) if ask_sizes else 0
                  mean_bid_size = np.mean(bid_sizes) if bid_sizes else 0
                  std_ask_size = np.std(ask_sizes) if ask_sizes else 0
                  std_bid_size = np.std(bid_sizes) if bid_sizes else 0

                  # Store extracted features
                  features.append({
                      'timestamp': ts,
                      'mid_price': mid_price,
                      'spread': spread,
                      'relative_spread': relative_spread,
                      'total_best_ask_volume': total_best_ask_volume,
                      'total_best_bid_volume': total_best_bid_volume,
                      'market_depth_ask': market_depth_ask,
                      'market_depth_bid': market_depth_bid,
                      'order_book_imbalance': order_book_imbalance,
                      'vwap_ask': vwap_ask,
                      'vwap_bid': vwap_bid,
                      'vwap_total': vwap_total,
                      'volume_imbalance_ratio': volume_imbalance_ratio,
                      'cumulative_delta_volume': cumulative_delta_volume,
                      'liquidity_pressure_ratio': liquidity_pressure_ratio,
                      'mean_ask_size': mean_ask_size,
                      'mean_bid_size': mean_bid_size,
                      'std_ask_size': std_ask_size,
                      'std_bid_size': std_bid_size
                  })

              except (json.JSONDecodeError, ValueError, IndexError) as e:
                  print(f'Skipping line due to error: {e}')

      # Convert to DataFrame
      return pd.DataFrame(features)

  # Calc features
  file_path = f'data/order_book/2025-02-17_{symbol}_ob500.data'
  df_features = process_order_book(file_path)

  # Convert timestamp to datetime
  df_features['timestamp'] = pd.to_datetime(df_features['timestamp'], unit='ms')  # Assuming milliseconds

  # Round timestamp to the nearest minute
  df_features['timestamp'] = df_features['timestamp'].dt.floor('min')

  # Aggregate order book features per minute
  agg_funcs = {
      'mid_price': ['mean', 'std', 'min', 'max', 'last'],  # Price statistics
      'spread': ['mean', 'std', 'max'],
      'relative_spread': ['mean', 'std'],
      'total_best_ask_volume': ['mean', 'std', 'sum', 'max'],
      'total_best_bid_volume': ['mean', 'std', 'sum', 'max'],
      'market_depth_ask': ['mean', 'std'],
      'market_depth_bid': ['mean', 'std'],
      'order_book_imbalance': ['mean', 'std'],
      'vwap_ask': ['mean'],
      'vwap_bid': ['mean'],
      'vwap_total': ['mean'],
      'volume_imbalance_ratio': ['mean', 'std'],
      'cumulative_delta_volume': ['last'],  # Running sum, take last value
      'liquidity_pressure_ratio': ['mean', 'std'],
      'mean_ask_size': ['mean', 'std'],
      'mean_bid_size': ['mean', 'std'],
      'std_ask_size': ['mean'],
      'std_bid_size': ['mean']
  }

  # Perform aggregation
  df_features_agg = df_features.groupby('timestamp').agg(agg_funcs)

  # Flatten multi-index column names
  df_features_agg.columns = ['_'.join(col).strip() for col in df_features_agg.columns]
  df_features_agg.reset_index(inplace=True)

  # Calculate Realized Volatility (1-min window): std of returns
  df_features_agg['realized_volatility'] = df_features_agg['mid_price_std'] / df_features_agg['mid_price_mean']

  # Convert startTime to timestamp
  df_features_agg['timestamp'] = df_features_agg['timestamp'].astype('int64') // 10**6

  # Save to CSV
  save_features(df_features_agg, 'features/orderbook', symbol)
  df_features_agg.to_csv('features/orderbook/features.csv', index=False)

def calc_features_funding_rate(symbol):
  df_features = load_data(f'data/funding_rate/{symbol}')

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

  # Convert startTime to timestamp
  df_features['fundingRateTimestamp'] = df_features['fundingRateTimestamp'].astype('int64') // 10**6

  save_features(df_features, 'features/funding_rate', symbol)

if __name__ == '__main__':
  calc_features_kline('BTCUSDT', '1')
  calc_features_order_book('BTCUSDT')
  calc_features_funding_rate('BTCUSDT')
