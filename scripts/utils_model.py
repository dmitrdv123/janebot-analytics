import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

def prepare_features_kline(df, prefix=None):
  df_scaled = df.copy()

  # Define columns for scale 
  columns_price = ['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10']
  columns_price_change = ['priceChange']
  columns_relative_price_change = ['relativePriceChange']
  columns_volume = ['volume']
  columns_turnover = ['turnover']
  columns_returns = ['logReturn', 'stdReturn_5m', 'stdReturn_10m', 'MACD_line', 'MACD_signal', 'MACD_histogram', 
             'Stochastic_K', 'Stochastic_D', 'ROC_14', 'RSI_14']
  columns_range = ['highLowRange']

  columns_price = columns_price if prefix == None else [f'{prefix}_{col}' for col in columns_price]
  columns_price_change = columns_price_change if prefix == None else [f'{prefix}_{col}' for col in columns_price_change]
  columns_relative_price_change = columns_relative_price_change if prefix == None else [f'{prefix}_{col}' for col in columns_relative_price_change]
  columns_volume = columns_volume if prefix == None else [f'{prefix}_{col}' for col in columns_volume]
  columns_turnover = columns_turnover if prefix == None else [f'{prefix}_{col}' for col in columns_turnover]
  columns_returns = columns_returns if prefix == None else [f'{prefix}_{col}' for col in columns_returns]
  columns_range = columns_range if prefix == None else [f'{prefix}_{col}' for col in columns_range]
  
  column_hourOfDay = 'hourOfDay' if prefix == None else f'{prefix}_hourOfDay'
  column_dayOfWeek = 'dayOfWeek' if prefix == None else f'{prefix}_dayOfWeek'
  column_weekOfYear = 'weekOfYear' if prefix == None else f'{prefix}_weekOfYear'
  column_monthOfYear = 'monthOfYear' if prefix == None else f'{prefix}_monthOfYear'
  column_minuteOfHour = 'minuteOfHour' if prefix == None else f'{prefix}_minuteOfHour'

  # Initialize scalers
  scalers = {
    'price': StandardScaler(),
    'price_change': RobustScaler(),
    'relative_price_change': StandardScaler(),
    'volume': RobustScaler(),
    'turnover': StandardScaler(),
    'returns': StandardScaler(),
    'range': StandardScaler()
  }

  # Apply different scalers
  df_scaled[columns_price] = scalers['price'].fit_transform(df_scaled[columns_price])
  df_scaled[columns_price_change] = scalers['price_change'].fit_transform(df_scaled[columns_price_change])
  df_scaled[columns_relative_price_change] = scalers['relative_price_change'].fit_transform(df_scaled[columns_relative_price_change])
  df_scaled[columns_volume] = scalers['volume'].fit_transform(df_scaled[columns_volume])
  df_scaled[columns_returns] = scalers['returns'].fit_transform(df_scaled[columns_returns])
  df_scaled[columns_range] = scalers['range'].fit_transform(df_scaled[columns_range])

  # Apply log transformation to turnover before scaling
  df_scaled[columns_turnover] = np.log1p(df_scaled[columns_turnover])  # log1p to avoid log(0) issues
  df_scaled[columns_turnover] = scalers['turnover'].fit_transform(df_scaled[columns_turnover])

  # Apply scale to categorical features
  df_scaled[column_hourOfDay] = df_scaled[column_hourOfDay] / 23  # Normalize to [0,1]
  df_scaled[column_dayOfWeek] = df_scaled[column_dayOfWeek] / 6  # Normalize to [0,1] (0=Monday, 6=Sunday)
  df_scaled[column_weekOfYear] = df_scaled[column_weekOfYear] / 51  # Normalize to [0,1]
  df_scaled[column_monthOfYear] = df_scaled[column_monthOfYear] / 11  # Normalize to [0,1] (0=Jan, 11=Dec)
  df_scaled[column_minuteOfHour] = df_scaled[column_minuteOfHour] / 59  # Normalize to [0,1]

  return df_scaled, scalers

def prepare_features_orderbook(df, prefix=None):
  df_scaled = df.copy()

  # Define columns for scale 
  columns_price = ['mid_price_mean', 'mid_price_min', 'mid_price_max', 'mid_price_first', 'mid_price_last', 'vwap_ask_mean', 'vwap_bid_mean', 'vwap_total_mean']
  columns_price_std = ['mid_price_std']
  columns_spread = ['spread_mean', 'spread_std']
  columns_spread_max = ['spread_max']
  columns_spread_relative = ['relative_spread_mean', 'relative_spread_std']
  columns_volume = ['total_best_ask_volume_mean', 'total_best_ask_volume_std', 'total_best_bid_volume_mean', 'total_best_bid_volume_std']
  columns_volume_max = ['total_best_ask_volume_max', 'total_best_bid_volume_max']
  columns_volume_sum = ['total_best_ask_volume_sum', 'total_best_bid_volume_sum']
  columns_volume_cumulative = ['cumulative_delta_volume_first', 'cumulative_delta_volume_last']
  columns_volume_ratio = ['order_flow_imbalance_mean', 'order_flow_imbalance_std', 'order_book_imbalance_mean', 'order_book_imbalance_std', 'volume_imbalance_ratio_mean', 'volume_imbalance_ratio_std', 'mean_ask_size_mean', 'mean_ask_size_std', 'mean_bid_size_mean', 'mean_bid_size_std', 'std_ask_size_mean', 'std_bid_size_mean']
  columns_market_depth = ['market_depth_ask_mean', 'market_depth_ask_std', 'market_depth_bid_mean', 'market_depth_bid_std']
  columns_liquidity = ['liquidity_pressure_ratio_mean', 'liquidity_pressure_ratio_std', 'depth_ratio_mean', 'depth_ratio_std']
  columns_count = ['price_level_count_ask_mean', 'price_level_count_bid_mean']
  columns_volatility = ['realized_volatility']

  columns_price = columns_price if prefix == None else [f'{prefix}_{col}' for col in columns_price]
  columns_price_std = columns_price_std if prefix == None else [f'{prefix}_{col}' for col in columns_price_std]
  columns_spread = columns_spread if prefix == None else [f'{prefix}_{col}' for col in columns_spread]
  columns_spread_max = columns_spread_max if prefix == None else [f'{prefix}_{col}' for col in columns_spread_max]
  columns_spread_relative = columns_spread_relative if prefix == None else [f'{prefix}_{col}' for col in columns_spread_relative]
  columns_volume = columns_volume if prefix == None else [f'{prefix}_{col}' for col in columns_volume]
  columns_volume_max = columns_volume_max if prefix == None else [f'{prefix}_{col}' for col in columns_volume_max]
  columns_volume_sum = columns_volume_sum if prefix == None else [f'{prefix}_{col}' for col in columns_volume_sum]
  columns_volume_cumulative = columns_volume_cumulative if prefix == None else [f'{prefix}_{col}' for col in columns_volume_cumulative]
  columns_volume_ratio = columns_volume_ratio if prefix == None else [f'{prefix}_{col}' for col in columns_volume_ratio]
  columns_market_depth = columns_market_depth if prefix == None else [f'{prefix}_{col}' for col in columns_market_depth]
  columns_liquidity = columns_liquidity if prefix == None else [f'{prefix}_{col}' for col in columns_liquidity]
  columns_count = columns_count if prefix == None else [f'{prefix}_{col}' for col in columns_count]
  columns_volatility = columns_volume_ratio if prefix == None else [f'{prefix}_{col}' for col in columns_volatility]

  # Initialize scalers
  scalers = {
    'price': StandardScaler(),
    'price_std': StandardScaler(),
    'spread': StandardScaler(),
    'spread_max': RobustScaler(),
    'spread_relative': StandardScaler(),
    'volume': RobustScaler(),# TODO
    'volume_max': RobustScaler(),
    'volume_sum': StandardScaler(),# TODO
    'volume_ratio': StandardScaler(),
    'volume_cumulative': StandardScaler(),
    'market_depth': StandardScaler(),# TODO
    'liquidity': StandardScaler(),
    'volatility': StandardScaler(),# TODO
    'count': StandardScaler(),# TODO
  }
  
  # Apply different scalers
  df_scaled[columns_price] = scalers['price'].fit_transform(df_scaled[columns_price])
  df_scaled[columns_price_std] = scalers['price_std'].fit_transform(df_scaled[columns_price_std])
  df_scaled[columns_spread] = scalers['spread'].fit_transform(df_scaled[columns_spread])
  df_scaled[columns_spread_max] = scalers['spread_max'].fit_transform(df_scaled[columns_spread_max])
  df_scaled[columns_spread_relative] = scalers['spread_relative'].fit_transform(df_scaled[columns_spread_relative])
  df_scaled[columns_volume_max] = scalers['volume_max'].fit_transform(df_scaled[columns_volume_max])
  df_scaled[columns_volume_ratio] = scalers['volume_ratio'].fit_transform(df_scaled[columns_volume_ratio])
  df_scaled[columns_volume_cumulative] = scalers['volume_cumulative'].fit_transform(df_scaled[columns_volume_cumulative])
  df_scaled[columns_liquidity] = scalers['liquidity'].fit_transform(df_scaled[columns_liquidity])
  df_scaled[columns_count] = scalers['count'].fit_transform(df_scaled[columns_count])
  
  # Apply log transformation to turnover before scaling
  df_scaled[columns_volume] = np.log1p(df_scaled[columns_volume])  # log1p to avoid log(0) issues
  df_scaled[columns_volume_sum] = np.log1p(df_scaled[columns_volume_sum])  # log1p to avoid log(0) issues
  df_scaled[columns_market_depth] = np.log1p(df_scaled[columns_market_depth])  # log1p to avoid log(0) issues
  df_scaled[columns_volatility] = np.log1p(df_scaled[columns_volatility])  # log1p to avoid log(0) issues
  
  df_scaled[columns_volume] = scalers['volume'].fit_transform(df_scaled[columns_volume])
  df_scaled[columns_volume_sum] = scalers['volume_sum'].fit_transform(df_scaled[columns_volume_sum])
  df_scaled[columns_market_depth] = scalers['market_depth'].fit_transform(df_scaled[columns_market_depth])
  df_scaled[columns_volatility] = scalers['volatility'].fit_transform(df_scaled[columns_volatility])
  
  return df_scaled, scalers

def prepare_features_long_short_ratio(df, prefix=None):
  df_scaled = df.copy()
  
  # Define columns for scale 
  columns_ratio = ['buyRatio', 'sellRatio']
  columns_other = ['net_position', 'long_short_ratio', 'sentiment_strength']
  columns_rolling = ['buyRatio_rolling_avg_15min', 'sellRatio_rolling_avg_15min', 'buyRatio_rolling_avg_30min', 'sellRatio_rolling_avg_30min', 'buyRatio_rolling_std_15min', 'sellRatio_rolling_std_15min', 'buyRatio_rolling_std_30min', 'sellRatio_rolling_std_30min']
  columns_roc = ['buyRatio_roc_5min', 'sellRatio_roc_5min', 'buyRatio_roc_15min', 'sellRatio_roc_15min', 'buyRatio_roc_30min', 'sellRatio_roc_30min']
  columns_lag = ['buyRatio_lag_1', 'sellRatio_lag_1']

  columns_ratio = columns_ratio if prefix == None else [f'{prefix}_{col}' for col in columns_ratio]
  columns_other = columns_other if prefix == None else [f'{prefix}_{col}' for col in columns_other]
  columns_rolling = columns_rolling if prefix == None else [f'{prefix}_{col}' for col in columns_rolling]
  columns_roc = columns_roc if prefix == None else [f'{prefix}_{col}' for col in columns_roc]
  columns_lag = columns_lag if prefix == None else [f'{prefix}_{col}' for col in columns_lag]

  # Initialize scalers
  scalers = {
    'ratio': MinMaxScaler(feature_range=(0, 1)),
    'other': StandardScaler(),
    'rolling': StandardScaler(),
    'roc': StandardScaler(),
    'lag': StandardScaler(),
  }
  
  # Apply different scalers
  df_scaled[columns_ratio] = scalers['ratio'].fit_transform(df_scaled[columns_ratio])
  df_scaled[columns_other] = scalers['other'].fit_transform(df_scaled[columns_other])
  df_scaled[columns_rolling] = scalers['rolling'].fit_transform(df_scaled[columns_rolling])
  df_scaled[columns_roc] = scalers['roc'].fit_transform(df_scaled[columns_roc])
  df_scaled[columns_lag] = scalers['lag'].fit_transform(df_scaled[columns_lag])

  return df_scaled, scalers

def prepare_features_funding_rate(df, prefix=None):
  df_scaled = df.copy()

  # Define columns for scale 
  columns_numeric = ['fundingRate', 'fundingRateMean_5m', 'fundingRateMean_10m']
  columns_volatility = ['fundingRateVolatility_5m', 'fundingRateVolatility_10m']
  columns_cumulative = ['cumulativeFundingRate']
  columns_change = ['fundingRateChange', 'fundingRateROC']
  columns_lag = ['fundingRateLag1']

  columns_numeric = columns_numeric if prefix == None else [f'{prefix}_{col}' for col in columns_numeric]
  columns_volatility = columns_volatility if prefix == None else [f'{prefix}_{col}' for col in columns_volatility]
  columns_cumulative = columns_cumulative if prefix == None else [f'{prefix}_{col}' for col in columns_cumulative]
  columns_change = columns_change if prefix == None else [f'{prefix}_{col}' for col in columns_change]
  columns_lag = columns_lag if prefix == None else [f'{prefix}_{col}' for col in columns_lag]

  # Initialize scalers
  scalers = {
    'numeric': MinMaxScaler(feature_range=(-1, 1)),
    'volatility': RobustScaler(),
    'cumulative': StandardScaler(),
    'change': StandardScaler(),
    'lag': MinMaxScaler(feature_range=(-1, 1)),
  }
  
  # Apply different scalers
  df_scaled[columns_numeric] = scalers['numeric'].fit_transform(df_scaled[columns_numeric])

  return df_scaled, scalers

def prepare_features_open_interest(df, prefix=None):
  df_scaled = df.copy()
  
  # Define columns for scale 
  columns_oi = ['openInterest', 'cumulative_open_interest']
  columns_rolling = ['oi_rolling_avg_15min', 'oi_rolling_avg_30min', 'oi_rolling_std_15min', 'oi_rolling_std_30min']
  columns_change = ['oi_pct_change', 'oi_rate_of_change_5min', 'cumulative_pct_change_oi', 'oi_momentum_15min', 'oi_momentum_30min']

  columns_oi = columns_oi if prefix == None else [f'{prefix}_{col}' for col in columns_oi]
  columns_rolling = columns_rolling if prefix == None else [f'{prefix}_{col}' for col in columns_rolling]
  columns_change = columns_change if prefix == None else [f'{prefix}_{col}' for col in columns_change]

  # Initialize scalers
  scalers = {
    'oi': RobustScaler(),
    'rolling': RobustScaler(),
    'change': StandardScaler(),
  }
  
  # Prepare features
  column_openInterest = 'openInterest' if prefix == None else f'{prefix}_openInterest'
  column_cumulative_open_interest = 'cumulative_open_interest' if prefix == None else f'{prefix}_cumulative_open_interest'
  column_oi_rolling_avg_15min = 'oi_rolling_avg_15min' if prefix == None else f'{prefix}_oi_rolling_avg_15min'
  column_oi_rolling_avg_30min = 'oi_rolling_avg_30min' if prefix == None else f'{prefix}_oi_rolling_avg_30min'
  column_oi_rolling_std_15min = 'oi_rolling_std_15min' if prefix == None else f'{prefix}_oi_rolling_std_15min'
  column_oi_rolling_std_30min = 'oi_rolling_std_30min' if prefix == None else f'{prefix}_oi_rolling_std_30min'
  
  column_oi_pct_change = 'oi_pct_change' if prefix == None else f'{prefix}_oi_pct_change'
  column_cumulative_pct_change_oi = 'cumulative_pct_change_oi' if prefix == None else f'{prefix}_cumulative_pct_change_oi'
  column_oi_momentum_15min = 'oi_momentum_15min' if prefix == None else f'{prefix}_oi_momentum_15min'
  column_oi_momentum_30min = 'oi_momentum_30min' if prefix == None else f'{prefix}_oi_momentum_30min'
  
  df_scaled[column_openInterest] = np.log1p(df_scaled[column_openInterest])
  df_scaled[column_cumulative_open_interest] = np.log1p(df_scaled[column_cumulative_open_interest])
  df_scaled[column_oi_rolling_avg_15min] = np.log1p(df_scaled[column_oi_rolling_avg_15min])
  df_scaled[column_oi_rolling_avg_30min] = np.log1p(df_scaled[column_oi_rolling_avg_30min])
  df_scaled[column_oi_rolling_std_15min] = np.log1p(df_scaled[column_oi_rolling_std_15min])
  df_scaled[column_oi_rolling_std_30min] = np.log1p(df_scaled[column_oi_rolling_std_30min])
  
  df_scaled[column_oi_pct_change] = df_scaled[column_oi_pct_change].replace([np.inf, -np.inf], np.nan).fillna(0)
  df_scaled[column_cumulative_pct_change_oi] = df_scaled[column_cumulative_pct_change_oi].replace([np.inf, -np.inf], np.nan).fillna(0)
  df_scaled[column_oi_momentum_15min] = df_scaled[column_oi_momentum_15min].replace([np.inf, -np.inf], np.nan).fillna(0)
  df_scaled[column_oi_momentum_30min] = df_scaled[column_oi_momentum_30min].replace([np.inf, -np.inf], np.nan).fillna(0)

  # Apply different scalers
  df_scaled[columns_oi] = scalers['oi'].fit_transform(df_scaled[columns_oi])
  df_scaled[columns_rolling] = scalers['rolling'].fit_transform(df_scaled[columns_rolling])
  df_scaled[columns_change] = scalers['change'].fit_transform(df_scaled[columns_change])

  return df_scaled, scalers

def merge_features(df_features1, df_features2, timestamp_col1, timestamp_col2, prefix1=None, prefix2=None):
  df_tmp1 = df_features1 if prefix1 == None else df_features1.add_prefix(f'{prefix1}_')
  df_tmp2 = df_features2 if prefix2 == None else df_features2.add_prefix(f'{prefix2}_')

  prefixed_timestamp_col1 = timestamp_col1 if prefix1 == None else f'{prefix1}_{timestamp_col1}'
  prefixed_timestamp_col2 = timestamp_col2 if prefix2 == None else f'{prefix2}_{timestamp_col2}'

  df_merged = pd.merge_asof(
    df_tmp1.sort_values(prefixed_timestamp_col1),
    df_tmp2.sort_values(prefixed_timestamp_col2),
    left_on=prefixed_timestamp_col1,
    right_on=prefixed_timestamp_col2,
    direction='backward'
  )

  return df_merged

def run_bot(df, y_pred, dataset_name, col_close_price='closePrice', signal_prob_threshold=None):
  position_amount_to_use = 10000

  fee_open = 0.0002
  fee_close = 0.0002

  long_signals = 0
  short_signals = 0

  long_correct = 0
  short_correct = 0

  total_profit = 0
  profits = []  # Track profit per trade

  for i in range(len(y_pred) - 1):
    current_row = df.iloc[i]
    next_row = df.iloc[i + 1]
    current_close = current_row[col_close_price]
    signal = np.argmax(y_pred[i])  # 0 (Hold), 1 (Short), 2 (Long)
    signal_prob = np.max(y_pred[i])
    
    if signal_prob_threshold != None and signal_prob < signal_prob_threshold:
      continue

    profit = 0.0
    if signal == 2:  # Long
      long_signals += 1
      next_price = next_row[col_close_price]

      position_amount = position_amount_to_use
      gross_profit = (next_price - current_close) / current_close * position_amount
      next_position_amount = position_amount + gross_profit
      open_fee = position_amount * fee_open
      close_fee = next_position_amount * fee_close
      profit = gross_profit - open_fee - close_fee

      stop_loss = -30
      if profit < stop_loss:
        profit = stop_loss - open_fee - close_fee

      if next_price > current_close:
        long_correct += 1
    elif signal == 1:  # Short
      short_signals += 1
      next_price = next_row[col_close_price]

      position_amount = position_amount_to_use
      gross_profit = (current_close - next_price) / current_close * position_amount
      next_position_amount = position_amount + gross_profit
      open_fee = position_amount * fee_open
      close_fee = next_position_amount * fee_close
      profit = gross_profit - open_fee - close_fee

      stop_loss = -30
      if profit < stop_loss:
        profit = stop_loss - open_fee - close_fee

      if next_price < current_close:
        short_correct += 1

    total_profit += profit
    profits.append(profit)

  total_signals = long_signals + short_signals
  total_correct = long_correct + short_correct
  accuracy = total_correct / total_signals * 100 if total_signals > 0 else 0
  avg_profit_per_trade = total_profit / total_signals if total_signals > 0 else 0
  
  # Calculate EV and RRR
  expected_value = 0.0
  reward_to_risk_ratio = 0.0
  if profits:
    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p < 0]
    
    avg_profit = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss = np.mean(losing_trades) if losing_trades else 0.0
    p_win = len(winning_trades) / len(profits) if profits else 0.0
    p_loss = 1 - p_win

    expected_value = (p_win * avg_profit) - (p_loss * abs(avg_loss))
    reward_to_risk_ratio = avg_profit / abs(avg_loss) if avg_loss != 0 else float('inf') if avg_profit > 0 else 0.0

  print(f"\nРезультаты бота для {dataset_name}:")
  print(f"Всего сигналов: {total_signals}")

  print(f"Сигналов Long: {long_signals}")
  print(f"Точных Long: {long_correct} ({long_correct/long_signals*100:.2f}% если >0)" if long_signals > 0 else "Точных Long: 0 (0%)")

  print(f"Сигналов Short: {short_signals}")
  print(f"Точных Short: {short_correct} ({short_correct/short_signals*100:.2f}% если >0)" if short_signals > 0 else "Точных Short: 0 (0%)")

  print(f"Общая точность: {accuracy:.2f}%")
  print(f"Общая прибыль/убыток: {total_profit:.2f} USD")
  print(f"Средняя прибыль на сделку: {avg_profit_per_trade:.2f} USD")

  print(f"Expected Value (EV): {expected_value:.4f} USD")
  print(f"Reward-to-Risk Ratio (RRR): {reward_to_risk_ratio:.4f}")

  return total_profit, accuracy
