import pandas as pd
import os
import time

from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv()
api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_SECRET_KEY")

client = HTTP(api_key=api_key, api_secret=api_secret)

def fetch_ohlcv(symbol, interval, start_time, end_time, limit):
    response = client.get_kline(symbol=symbol, interval=interval, start=start_time, end=end_time, limit=limit)

    if response['retCode'] != 0:
        raise Exception(f"Error fetching kline data: {response['retMsg']}")

    data = response['result']['list']
    return data

def download_data(symbol, interval, start_time, end_time):
    # Loop through each hour of the day (or however long the period is)
    while start_time < end_time:
        # Set the end_time for the current chunk, ensuring no overlap
        hour_end_time = min(start_time + (60 * 60 * 1000) - 1, end_time)  # Subtract 1 ms to avoid overlap

        # Fetch the data for this 1-hour chunk
        print(f"Fetching data from {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time/1000))} to {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(hour_end_time/1000))}")
        data = fetch_ohlcv(symbol, interval, start_time, hour_end_time, 60 // int(interval))

        if data:
            df = pd.DataFrame(data, columns=['startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover'])
            df = df.sort_values('startTime', ascending=True)
            df.to_csv(f"data/kline/{symbol}_{interval}_{start_time}_{end_time}.csv", index=False)
            print(f"Saved: data/kline/{symbol}_{interval}_{start_time}_{end_time}.csv")
        else:
            break

        # Move the start_time to the next chunk (next hour)
        start_time = hour_end_time + 1  # Start the next chunk right after the current chunk's end time

        # Sleep to avoid hitting API rate limits
        time.sleep(1)

symbol = "BTCUSDT"
interval = "1"  # 1-minute candles
end_time = int(time.time() * 1000)  # Current timestamp in milliseconds
start_time = end_time - (48 * 60 * 60 * 1000)  # N hours ago

download_data(symbol, interval, start_time, end_time)
