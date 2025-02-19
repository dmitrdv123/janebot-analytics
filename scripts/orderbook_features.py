import json
import pandas as pd
import numpy as np

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
file_path = 'data/orderbook/2025-02-17_BTCUSDT_ob500.data'
df_features = process_order_book(file_path)

# Convert timestamp to datetime
df_features['timestamp'] = pd.to_datetime(df_features['timestamp'], unit='ms')  # Assuming milliseconds

# Round timestamp to the nearest minute
df_features['timestamp'] = df_features['timestamp'].dt.floor('min')  
df_features['timestamp'] = df_features['timestamp'].astype('int64') // 10**6

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
    'std_bid_size': ['mean'],
    'best_bid_price': ['mean', 'std'],
    'best_ask_price': ['mean', 'std'],
    'mean_bid_price': ['mean'],
    'mean_ask_price': ['mean'],
    'mean_mid_price': ['mean'],
    'std_bid_price': ['std'],
    'std_ask_price': ['std'],
    'std_mid_price': ['std']
}

# Perform aggregation
df_features_agg = df_features.groupby('timestamp').agg(agg_funcs)

# Flatten multi-index column names
df_features_agg.columns = ['_'.join(col).strip() for col in df_features_agg.columns]
df_features_agg.reset_index(inplace=True)

# Print sample of feature dataframe
print(df_features_agg.head())

# Save to CSV
df_features_agg.to_csv('features/orderbook/features.csv', index=False)
