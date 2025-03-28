import os
import pandas as pd
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP

from utils import ensure_directory

# Initialize Bybit API (No authentication needed for public data)
session = HTTP()

# Function to split time range into 1-hour chunks
def split_time_range(start_time, end_time, hours=1):
  chunks = []
  while start_time < end_time:
    next_time = min(start_time + timedelta(hours=hours), end_time)
    chunk_end = next_time - timedelta(milliseconds=1) if next_time < end_time else end_time
    chunks.append((start_time, chunk_end))
    start_time = next_time
  return chunks

# Function to save data per hour inside the correct date-based folder
def save_data(df, base_folder, symbol, start_time, interval=None):
  if interval:
    date_folder = f'{base_folder}/{symbol}/{interval}/{start_time.strftime("%Y-%m-%d")}'
  else:
    date_folder = f'{base_folder}/{symbol}/{start_time.strftime("%Y-%m-%d")}'
  ensure_directory(date_folder)
  filename = f'{date_folder}/{start_time.strftime("%H")}.csv'
  df.to_csv(filename, index=False)
  print(f'Data saved to {filename}')

# Function to download Kline data
def download_kline(symbol, interval, start_time, end_time):
  base_folder = 'data/kline'

  time_chunks = split_time_range(start_time, end_time)
  for start, end in time_chunks:
    response = session.get_kline(
      category='linear',
      symbol=symbol,
      interval=interval,
      start=int(start.timestamp() * 1000),
      end=int(end.timestamp() * 1000),
      limit=1000
    )
    if 'result' in response and 'list' in response['result']:
      data = response['result']['list']

      df = pd.DataFrame(data, columns=['startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover'])
      df = df.sort_values('startTime', ascending=True)

      save_data(df, base_folder, symbol, start, interval)

# Function to download Funding Rate history
def download_funding_rate(symbol, start_time, end_time):
  base_folder = 'data/funding_rate'

  time_chunks = split_time_range(start_time, end_time, hours=24)
  for start, end in time_chunks:
    response = session.get_funding_rate_history(
      category='linear',
      symbol=symbol,
      startTime=int(start.timestamp() * 1000),
      endTime=int(end.timestamp() * 1000),
      limit=1000
    )
    if 'result' in response and 'list' in response['result']:
      data = response['result']['list']

      df = pd.DataFrame(data, columns=['symbol', 'fundingRate', 'fundingRateTimestamp'])
      df = df.sort_values('fundingRateTimestamp', ascending=True)

      save_data(df, base_folder, symbol, start)

# Function to download Long/Short Ratio
def download_long_short_ratio(symbol, period, start_time, end_time):
  base_folder = 'data/long_short_ratio'

  time_chunks = split_time_range(start_time, end_time)
  for start, end in time_chunks:
    cursor = None
    data = []
    while True:
      response = session.get_long_short_ratio(
        category='linear',
        symbol=symbol,
        period=period,
        startTime=int(start.timestamp() * 1000),
        endTime=int(end.timestamp() * 1000),
        limit=1000,
        cursor=cursor
      )
      if 'result' in response and 'list' in response['result']:
        data.extend(response['result']['list'])
        cursor = response.get('nextPageCursor')

        if not cursor:
          break

    df = pd.DataFrame(data, columns=['symbol', 'buyRatio', 'sellRatio', 'timestamp'])
    df = df.sort_values('timestamp', ascending=True)

    save_data(df, base_folder, symbol, start, period)

# Function to download Open Interest
def download_open_interest(symbol, intervalTime, start_time, end_time):
  base_folder = 'data/open_interest'

  time_chunks = split_time_range(start_time, end_time)
  for start, end in time_chunks:
    cursor = None
    data = []
    while True:
      response = session.get_open_interest(
        category='linear',
        symbol=symbol,
        intervalTime=intervalTime,
        startTime=int(start.timestamp() * 1000),
        endTime=int(end.timestamp() * 1000),
        limit=1000,
        cursor=cursor
      )
      if 'result' in response and 'list' in response['result']:
        data.extend(response['result']['list'])
        cursor = response.get('nextPageCursor')

        if not cursor:
          break

    df = pd.DataFrame(data, columns=['openInterest', 'timestamp'])
    df = df.sort_values('timestamp', ascending=True)

    save_data(df, base_folder, symbol, start, intervalTime)

# Function to download Index Price Kline
def download_index_price_kline(symbol, interval, start_time, end_time):
  base_folder = 'data/index_price_kline'

  time_chunks = split_time_range(start_time, end_time)
  for start, end in time_chunks:
    response = session.get_index_price_kline(
      category='linear',
      symbol=symbol,
      interval=interval,
      startTime=int(start.timestamp() * 1000),
      endTime=int(end.timestamp() * 1000),
      limit=1000
    )
    if 'result' in response and 'list' in response['result']:
      data = response['result']['list']

      df = pd.DataFrame(data, columns=['startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice'])
      df = df.sort_values('startTime', ascending=True)

      save_data(df, base_folder, symbol, start, interval)

# Function to download Mark Price Kline
def download_mark_price_kline(symbol, interval, start_time, end_time):
  base_folder = 'data/mark_price_kline'

  time_chunks = split_time_range(start_time, end_time)
  for start, end in time_chunks:
    response = session.get_mark_price_kline(
      category='linear',
      symbol=symbol,
      interval=interval,
      startTime=int(start.timestamp() * 1000),
      endTime=int(end.timestamp() * 1000),
      limit=1000
    )
    if 'result' in response and 'list' in response['result']:
      data = response['result']['list']

      df = pd.DataFrame(data, columns=['startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice'])
      df = df.sort_values('startTime', ascending=True)

      save_data(df, base_folder, symbol, start, interval)

# Function to download Premium Index Price Kline
def download_premium_index_price_kline(symbol, interval, start_time, end_time):
  base_folder = 'data/premium_index_price_kline'

  time_chunks = split_time_range(start_time, end_time)
  for start, end in time_chunks:
    response = session.get_premium_index_price_kline(
      category='linear',
      symbol=symbol,
      interval=interval,
      startTime=int(start.timestamp() * 1000),
      endTime=int(end.timestamp() * 1000),
      limit=1000
    )
    if 'result' in response and 'list' in response['result']:
      data = response['result']['list']

      df = pd.DataFrame(data, columns=['startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice'])
      df = df.sort_values('startTime', ascending=True)

      save_data(df, base_folder, symbol, start, interval)

# Example usage
if __name__ == '__main__':
  symbol = 'BROCCOLIUSDT'
  interval = '5'  # Kline interval (1m, 5m, 15m, etc.)
  period_long_short_ratio = '5min'  # Period for Long/Short Ratio (5min 15min 30min 1h 4h 4d)
  intervalTime = '5min'  # Interval Time for Open Interest (5min 15min 30min 1h 4h 1d)
  start_time = datetime(2025, 3, 23)
  end_time = datetime(2025, 3, 28)

  download_kline(symbol, interval, start_time, end_time)
  download_funding_rate(symbol, start_time, end_time)
  # download_long_short_ratio(symbol, period_long_short_ratio, start_time, end_time)
  # download_open_interest(symbol, intervalTime, start_time, end_time)
  # download_index_price_kline(symbol, interval, start_time, end_time)
  # download_mark_price_kline(symbol, interval, start_time, end_time)
  # download_premium_index_price_kline(symbol, interval, start_time, end_time)
