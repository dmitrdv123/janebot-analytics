# Define feature columns
columns_input_kline = [
  'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover',
  'priceChange', 'relativePriceChange', 'logReturn', 'SMA_5', 'SMA_10',
  'EMA_5', 'EMA_10', 'hourOfDay', 'dayOfWeek', 'weekOfYear', 'monthOfYear',
  'minuteOfHour', 'isWeekend', 'highLowRange', 'stdReturn_5m', 'stdReturn_10m',
  'RSI_14', 'MACD_line', 'MACD_signal', 'MACD_histogram', 'Stochastic_K',
  'Stochastic_D', 'ROC_14'
]

columns_input_orderbook = [
  'mid_price_mean', 'mid_price_std', 'mid_price_min', 'mid_price_max', 'mid_price_first', 'mid_price_last', 
  'spread_mean', 'spread_std', 'spread_max', 'relative_spread_mean', 'relative_spread_std', 
  'total_best_ask_volume_mean', 'total_best_ask_volume_std', 'total_best_ask_volume_sum', 'total_best_ask_volume_max', 
  'total_best_bid_volume_mean', 'total_best_bid_volume_std', 'total_best_bid_volume_sum', 'total_best_bid_volume_max', 
  'market_depth_ask_mean', 'market_depth_ask_std', 'market_depth_bid_mean', 'market_depth_bid_std', 'order_book_imbalance_mean', 
  'order_book_imbalance_std', 'vwap_ask_mean', 'vwap_bid_mean', 'vwap_total_mean', 'volume_imbalance_ratio_mean', 
  'volume_imbalance_ratio_std', 'cumulative_delta_volume_first', 'cumulative_delta_volume_last', 'liquidity_pressure_ratio_mean', 'liquidity_pressure_ratio_std', 
  'mean_ask_size_mean', 'mean_ask_size_std', 'mean_bid_size_mean', 'mean_bid_size_std', 'std_ask_size_mean', 'std_bid_size_mean', 
  'order_flow_imbalance_mean', 'order_flow_imbalance_std', 'depth_ratio_mean', 'depth_ratio_std', 'price_level_count_ask_mean', 
  'price_level_count_bid_mean', 'realized_volatility',
]

columns_input_long_short_ratio = [
  'buyRatio', 'sellRatio', 'net_position', 'long_short_ratio', 'buyRatio_rolling_avg_15min', 'sellRatio_rolling_avg_15min', 
  'buyRatio_rolling_avg_30min', 'sellRatio_rolling_avg_30min', 'buyRatio_rolling_std_15min', 'sellRatio_rolling_std_15min', 
  'buyRatio_rolling_std_30min', 'sellRatio_rolling_std_30min', 'buyRatio_roc_5min', 'sellRatio_roc_5min', 'buyRatio_roc_15min', 
  'sellRatio_roc_15min', 'buyRatio_roc_30min', 'sellRatio_roc_30min', 'sentiment_class', 'sentiment_strength', 'buyRatio_lag_1', 
  'sellRatio_lag_1', 'extreme_buy_ratio', 'extreme_sell_ratio'
]

columns_input_funding_rate = [
  'fundingRate', 'fundingRateMean_5m', 'fundingRateMean_10m', 
  'fundingRateVolatility_5m', 'fundingRateVolatility_10m', 
  'cumulativeFundingRate', 'fundingRateChange', 'fundingRateROC', 
  'sentiment', 'fundingRateLag1'
]

columns_open_interest = [
  'openInterest', 'oi_pct_change', 'oi_rate_of_change_5min', 'oi_sentiment',
  'oi_rolling_avg_15min', 'oi_rolling_std_15min', 
  'oi_rolling_avg_30min', 'oi_rolling_std_30min', 
  'oi_momentum_15min', 'oi_momentum_30min', 
  'cumulative_open_interest', 'cumulative_pct_change_oi', 
]
