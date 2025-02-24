import json
import os
import numpy as np
import pandas as pd

from utils import calculate_macd, calculate_roc, calculate_rsi, calculate_stochastic, ensure_directory

def load_data(base_folder_path: str, file_extension: str = '.csv'):
  '''
  Downloads all files from subfolders (organized by date) inside a given folder and combines them into a single dataset (DataFrame).

  Args:
  - base_folder_path (str): Path to the base folder containing subfolders (named by date).
  - file_extension (str): Extension of files to download (default is '.csv').

  Returns:
  - pd.DataFrame: Combined dataset of all files in the subfolders.
  '''
  all_files = []

  # Check if base folder exists
  if not os.path.exists(base_folder_path):
    raise FileNotFoundError(f'The base folder {base_folder_path} does not exist.')

  # Iterate over the subfolders (which represent dates)
  for subfolder_name in os.listdir(base_folder_path):
    subfolder_path = os.path.join(base_folder_path, subfolder_name)

    # Check if the subfolder is indeed a directory (i.e., not a file)
    if os.path.isdir(subfolder_path):
      # Iterate over the files in the subfolder
      for filename in os.listdir(subfolder_path):
        # Filter based on file extension (e.g., CSV)
        if filename.endswith(file_extension):
          file_path = os.path.join(subfolder_path, filename)
          # Read file (assuming CSV for simplicity)
          df = pd.read_csv(file_path)
          all_files.append(df)

  # Combine all files into a single DataFrame
  if all_files:
    combined_data = pd.concat(all_files, ignore_index=True)
    return combined_data
  else:
    raise ValueError('No files found in the subfolders with the specified extension.')

def save_features(df, base_folder, symbol, interval=None):
  if interval:
    date_folder = f'{base_folder}/{symbol}/{interval}'
  else:
    date_folder = f'{base_folder}/{symbol}'
  ensure_directory(date_folder)
  filename = f'{date_folder}/features.csv'
  df.to_csv(filename, index=False)
  print(f'Feature saved to {filename}')

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

def calc_features_kline(symbol, interval):
  print(f'Calculate kline features')

  # Load the Kline Data
  df_features = load_data(f'data/kline/{symbol}/{interval}')

  # Ensure numeric columns are properly cast
  df_features['volume'] = df_features['volume'].astype(float)
  df_features['turnover'] = df_features['turnover'].astype(float)

  df_features = calc_features_kline_based(df_features)
  
  # Order by timestamp ascending
  df_features = df_features.sort_values(by='startTime')

  # Save the features to a CSV file
  save_features(df_features, 'features/kline', symbol, interval)

def calc_features_index_price_kline(symbol, interval):
  print(f'Calculate index price kline features')

  # Load the Kline Data
  df_features = load_data(f'data/index_price_kline/{symbol}/{interval}')

  df_features = calc_features_kline_based(df_features)

  # Save the features to a CSV file
  save_features(df_features, 'features/index_price_kline', symbol, interval)

def calc_features_mark_price_kline(symbol, interval):
  print(f'Calculate mark price kline features')

  # Load the Kline Data
  df_features = load_data(f'data/mark_price_kline/{symbol}/{interval}')

  df_features = calc_features_kline_based(df_features)

  # Save the features to a CSV file
  save_features(df_features, 'features/mark_price_kline', symbol, interval)

def calc_features_premium_index_price_kline(symbol, interval):
  print(f'Calculate premium index price kline features')

  # Load the Kline Data
  df_features = load_data(f'data/premium_index_price_kline/{symbol}/{interval}')

  df_features = calc_features_kline_based(df_features)

  # Save the features to a CSV file
  save_features(df_features, 'features/premium_index_price_kline', symbol, interval)

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
                  liquidity_pressure_ratio = np.log1p(total_best_bid_volume) - np.log1p(total_best_ask_volume)

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

  print(f'Calculate order book features')

  all_features = []

  # Iterate over all files in the order book directory
  order_book_dir = f'data/order_book/{symbol}'
  for filename in os.listdir(order_book_dir):
      file_path = os.path.join(order_book_dir, filename)
      if os.path.isfile(file_path):
          df_features = process_order_book(file_path)
          all_features.append(df_features)

  # Combine all features into a single DataFrame
  df_features = pd.concat(all_features, ignore_index=True)

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

  # Order by timestamp ascending
  df_features_agg = df_features_agg.sort_values(by='timestamp')

  # Save to CSV
  save_features(df_features_agg, 'features/order_book', symbol)

def calc_features_funding_rate(symbol):
  print(f'Calculate funding rate features')

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
  df_features['sentiment'] = df_features['fundingRate'].apply(lambda x: '1' if x > 0 else '-1')

  # Lagged feature
  df_features['fundingRateLag1'] = df_features['fundingRate'].shift(1)

  # Convert startTime to timestamp
  df_features['fundingRateTimestamp'] = df_features['fundingRateTimestamp'].astype('int64') // 10**6
  
  # Order by timestamp ascending
  df_features = df_features.sort_values(by='fundingRateTimestamp')

  save_features(df_features, 'features/funding_rate', symbol)

def calc_features_long_short_ratio(symbol, period):
  print(f'Calculate long short ratio features')

  df_features = load_data(f'data/long_short_ratio/{symbol}/{period}')

  # Convert timestamp to datetime
  df_features['timestamp'] = pd.to_datetime(df_features['timestamp'], unit='ms')

  # Ensure numeric columns are properly cast
  df_features['buyRatio'] = df_features['buyRatio'].astype(float)
  df_features['sellRatio'] = df_features['sellRatio'].astype(float)

  # Features

  # Calculate Net Long/Short Position
  df_features['net_position'] = df_features['buyRatio'] - df_features['sellRatio']

  # Calculate Long/Short Ratio
  df_features['long_short_ratio'] = df_features['buyRatio'] / df_features['sellRatio']

  # Rolling average over 15 minutes
  df_features['buyRatio_rolling_avg_15min'] = df_features['buyRatio'].rolling(window=3).mean()
  df_features['sellRatio_rolling_avg_15min'] = df_features['sellRatio'].rolling(window=3).mean()

  df_features['buyRatio_rolling_avg_30min'] = df_features['buyRatio'].rolling(window=6).mean()
  df_features['sellRatio_rolling_avg_30min'] = df_features['sellRatio'].rolling(window=6).mean()

  # Rolling standard deviation over 15 minutes
  df_features['buyRatio_rolling_std_15min'] = df_features['buyRatio'].rolling(window=3).std()
  df_features['sellRatio_rolling_std_15min'] = df_features['sellRatio'].rolling(window=3).std()

  df_features['buyRatio_rolling_std_30min'] = df_features['buyRatio'].rolling(window=6).std()
  df_features['sellRatio_rolling_std_30min'] = df_features['sellRatio'].rolling(window=6).std()

  # Rate of Change for buy and sell ratios
  df_features['buyRatio_roc_5min'] = df_features['buyRatio'].pct_change(periods=1)
  df_features['sellRatio_roc_5min'] = df_features['sellRatio'].pct_change(periods=1)

  df_features['buyRatio_roc_15min'] = df_features['buyRatio'].pct_change(periods=3)
  df_features['sellRatio_roc_15min'] = df_features['sellRatio'].pct_change(periods=3)

  df_features['buyRatio_roc_30min'] = df_features['buyRatio'].pct_change(periods=6)
  df_features['sellRatio_roc_30min'] = df_features['sellRatio'].pct_change(periods=6)

  # Sentiment Classification: Bullish or Bearish
  df_features['sentiment_class'] = np.where(df_features['net_position'] > 0, 1, -1)

  # Sentiment Strength (absolute value of net position)
  df_features['sentiment_strength'] = df_features['net_position'].abs()

  # Add lag features (1-minute lag example)
  df_features['buyRatio_lag_1'] = df_features['buyRatio'].shift(1)
  df_features['sellRatio_lag_1'] = df_features['sellRatio'].shift(1)

  # Extreme buy or sell ratio
  df_features['extreme_buy_ratio'] = df_features['buyRatio'] > 0.9  # Threshold for extreme sentiment
  df_features['extreme_sell_ratio'] = df_features['sellRatio'] > 0.9

  # Convert startTime to timestamp
  df_features['timestamp'] = df_features['timestamp'].astype('int64') // 10**6
  
  # Order by timestamp ascending
  df_features = df_features.sort_values(by='timestamp')

  save_features(df_features, f'features/long_short_ratio', symbol, period)

def calc_features_open_interest(symbol, intervalTime):
  print(f'Calculate open interest features')

  df_features = load_data(f'data/open_interest/{symbol}/{intervalTime}')

  # Convert timestamp to datetime
  df_features['timestamp'] = pd.to_datetime(df_features['timestamp'], unit='ms')

  # Ensure numeric columns are properly cast
  df_features['openInterest'] = df_features['openInterest'].astype(float)

  # Features

  # Rolling averages and standard deviations
  df_features['oi_rolling_avg_15min'] = df_features['openInterest'].rolling(window=3).mean()
  df_features['oi_rolling_std_15min'] = df_features['openInterest'].rolling(window=3).std()

  df_features['oi_rolling_avg_30min'] = df_features['openInterest'].rolling(window=6).mean()
  df_features['oi_rolling_std_30min'] = df_features['openInterest'].rolling(window=6).std()

  # Rate of change and percentage change
  df_features['oi_pct_change'] = df_features['openInterest'].pct_change()
  df_features['oi_rate_of_change_5min'] = df_features['openInterest'].diff()

  # Cumulative Open Interest and Cumulative Percentage Change
  df_features['cumulative_open_interest'] = df_features['openInterest'].cumsum()
  df_features['cumulative_pct_change_oi'] = df_features['oi_pct_change'].cumsum()

  # Momentum of Open Interest
  df_features['oi_momentum_15min'] = df_features['oi_pct_change'].rolling(window=3).mean()
  df_features['oi_momentum_30min'] = df_features['oi_pct_change'].rolling(window=6).mean()

  # Sentiment of Open Interest (for example, positive if Open Interest is increasing)
  df_features['oi_sentiment'] = np.where(df_features['oi_pct_change'] > 0, 1, -1)

  # Convert startTime to timestamp
  df_features['timestamp'] = df_features['timestamp'].astype('int64') // 10**6
  
  # Order by timestamp ascending
  df_features = df_features.sort_values(by='timestamp')

  save_features(df_features, f'features/open_interest', symbol, intervalTime)

def merge_features_open_interest(symbol, interval, intervalTime, period_long_short_ratio):
  df_features_kline = pd.read_csv(f'features/kline/{symbol}/{interval}/features.csv')
  df_features_mark_price_kline = pd.read_csv(f'features/mark_price_kline/{symbol}/{interval}/features.csv')
  df_features_index_price_kline = pd.read_csv(f'features/index_price_kline/{symbol}/{interval}/features.csv')
  df_features_premium_index_price_kline = pd.read_csv(f'features/premium_index_price_kline/{symbol}/{interval}/features.csv')
  
  df_features_funding_rate = pd.read_csv(f'features/funding_rate/{symbol}/features.csv')
  df_features_long_short_ratio = pd.read_csv(f'features/long_short_ratio/{symbol}/{period_long_short_ratio}/features.csv')
  df_features_open_interest = pd.read_csv(f'features/open_interest/{symbol}/{intervalTime}/features.csv')
  df_features_order_book = pd.read_csv(f'features/order_book/{symbol}/features.csv')
  
  df_features_kline = df_features_kline.add_prefix('features_kline_')
  df_features_mark_price_kline = df_features_mark_price_kline.add_prefix('features_mark_price_kline_')
  df_features_index_price_kline = df_features_index_price_kline.add_prefix('features_index_price_kline_')
  df_features_premium_index_price_kline = df_features_premium_index_price_kline.add_prefix('features_premium_index_price_kline_')

  df_features_funding_rate = df_features_funding_rate.add_prefix('features_funding_rate_')
  df_features_long_short_ratio = df_features_long_short_ratio.add_prefix('features_long_short_ratio_')
  df_features_open_interest = df_features_open_interest.add_prefix('features_open_interest_')
  df_features_order_book = df_features_order_book.add_prefix('features_order_book_')
  
  df_merged = pd.merge_asof(
    df_features_kline.sort_values('features_kline_startTime'),
    df_features_mark_price_kline.sort_values('features_mark_price_kline_startTime'),
    left_on='features_kline_startTime',
    right_on='features_mark_price_kline_startTime',
    direction='backward'
  )
  
  df_merged = pd.merge_asof(
    df_merged.sort_values('features_kline_startTime'),
    df_features_index_price_kline.sort_values('features_index_price_kline_startTime'),
    left_on='features_kline_startTime',
    right_on='features_index_price_kline_startTime',
    direction='backward'
  )
  
  df_merged = pd.merge_asof(
    df_merged.sort_values('features_kline_startTime'),
    df_features_premium_index_price_kline.sort_values('features_premium_index_price_kline_startTime'),
    left_on='features_kline_startTime',
    right_on='features_premium_index_price_kline_startTime',
    direction='backward'
  )
  
  df_merged = pd.merge_asof(
    df_merged.sort_values('features_kline_startTime'),
    df_features_funding_rate.sort_values('features_funding_rate_fundingRateTimestamp'),
    left_on='features_kline_startTime',
    right_on='features_funding_rate_fundingRateTimestamp',
    direction='backward'
  )
  
  df_merged = pd.merge_asof(
    df_merged.sort_values('features_kline_startTime'),
    df_features_long_short_ratio.sort_values('features_long_short_ratio_timestamp'),
    left_on='features_kline_startTime',
    right_on='features_long_short_ratio_timestamp',
    direction='backward'
  )
  
  df_merged = pd.merge_asof(
    df_merged.sort_values('features_kline_startTime'),
    df_features_open_interest.sort_values('features_open_interest_timestamp'),
    left_on='features_kline_startTime',
    right_on='features_open_interest_timestamp',
    direction='backward'
  )
  
  df_merged = pd.merge_asof(
    df_merged.sort_values('features_kline_startTime'),
    df_features_order_book.sort_values('features_order_book_timestamp'),
    left_on='features_kline_startTime',
    right_on='features_order_book_timestamp',
    direction='backward'
  )
  
  # Filter by order book interval
  min_timestamp = df_features_order_book['features_order_book_timestamp'].min()
  max_timestamp = df_features_order_book['features_order_book_timestamp'].max()
  df_merged = df_merged[(df_merged['features_kline_startTime'] >= min_timestamp) & (df_merged['features_kline_startTime'] <= max_timestamp)]

  return df_merged

def calc_features_merged(symbol, interval, intervalTime, period_long_short_ratio):
  df = merge_features_open_interest(symbol, interval, intervalTime, period_long_short_ratio)
  
  # Open Interest to Price ratio
  df['features_open_interest_oi_to_price_ratio'] = df['features_open_interest_openInterest'] / df['features_kline_closePrice']
  
  # Price vs Open Interest Divergence
  df['features_open_interest_price_vs_oi_divergence'] = df['features_kline_closePrice'].pct_change() - df['features_open_interest_openInterest'].pct_change()
  
  # Calculate correlation between buyRatio and price over 5 minutes
  df['features_long_short_ratio_corr_long_short_price_5min'] = df['features_long_short_ratio_buyRatio'].rolling(window=5).corr(df['features_kline_closePrice'])
  df['features_long_short_ratio_corr_long_short_price_10min'] = df['features_long_short_ratio_buyRatio'].rolling(window=10).corr(df['features_kline_closePrice'])

  # Order by timestamp ascending
  df = df.sort_values(by='features_kline_startTime')

  save_features(df, f'features/merged', symbol)

if __name__ == '__main__':
  symbol = 'BTCUSDT'
  interval = '1'  # Kline interval (1m, 5m, 15m, etc.)
  intervalTime = '5min'  # Interval Time for Open Interest (5min 15min 30min 1h 4h 1d)
  period_long_short_ratio = '5min'  # Period for Long/Short Ratio (5min 15min 30min 1h 4h 4d)

  calc_features_kline(symbol, interval)
  calc_features_index_price_kline(symbol, interval)
  calc_features_mark_price_kline(symbol, interval)
  calc_features_premium_index_price_kline(symbol, interval)
  calc_features_order_book(symbol)
  calc_features_funding_rate(symbol)
  calc_features_long_short_ratio(symbol, period_long_short_ratio)
  calc_features_open_interest(symbol, intervalTime)

  calc_features_merged(symbol, interval, intervalTime, period_long_short_ratio)
