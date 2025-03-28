import random
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable
from sklearn.preprocessing import RobustScaler, StandardScaler

def scale_features(df, scalers=None):
  df_scaled = df.copy()

  # Define columns for scale
  price_columns = ['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'SMA_short', 'SMA_long', 'EMA_short', 'EMA_long']
  price_change_column = ['priceChange', 'priceChange_window_short', 'priceChange_window_long']
  relative_price_change_column = ['relativePriceChange', 'relativePriceChange_window_short', 'relativePriceChange_window_long']
  volume_columns = ['volume']
  turnover_column = ['turnover']
  returns_columns = ['logReturn', 'stdReturn_short', 'stdReturn_long', 'MACD_line', 'MACD_signal', 'MACD_histogram', 
             'Stochastic_K_short', 'Stochastic_D_short', 'Stochastic_K_long', 'Stochastic_D_long', 'ROC_short', 'ROC_long', 'RSI_short', 'RSI_long']
  range_columns = ['highLowRange']

  if not scalers:
    scalers = {
      'price': StandardScaler(),
      'price_change': RobustScaler(),
      'relative_price_change': StandardScaler(),
      'volume': RobustScaler(),
      'turnover': StandardScaler(),
      'returns': StandardScaler(),
      'range': StandardScaler(),
    }

    # Apply different scalers
    df_scaled[price_columns] = scalers['price'].fit_transform(df_scaled[price_columns])
    df_scaled[price_change_column] = scalers['price_change'].fit_transform(df_scaled[price_change_column])
    df_scaled[relative_price_change_column] = scalers['relative_price_change'].fit_transform(df_scaled[relative_price_change_column])
    df_scaled[volume_columns] = scalers['volume'].fit_transform(df_scaled[volume_columns])
    df_scaled[returns_columns] = scalers['returns'].fit_transform(df_scaled[returns_columns])
    df_scaled[range_columns] = scalers['range'].fit_transform(df_scaled[range_columns])

    # Apply log transformation to turnover before scaling
    df_scaled[turnover_column] = np.log1p(df_scaled[turnover_column])  # log1p to avoid log(0) issues
    df_scaled[turnover_column] = scalers['turnover'].fit_transform(df_scaled[turnover_column])
  else:
    # Apply different scalers
    df_scaled[price_columns] = scalers['price'].transform(df_scaled[price_columns])
    df_scaled[price_change_column] = scalers['price_change'].transform(df_scaled[price_change_column])
    df_scaled[relative_price_change_column] = scalers['relative_price_change'].transform(df_scaled[relative_price_change_column])
    df_scaled[volume_columns] = scalers['volume'].transform(df_scaled[volume_columns])
    df_scaled[returns_columns] = scalers['returns'].transform(df_scaled[returns_columns])
    df_scaled[range_columns] = scalers['range'].transform(df_scaled[range_columns])

    # Apply log transformation to turnover before scaling
    df_scaled[turnover_column] = np.log1p(df_scaled[turnover_column])  # log1p to avoid log(0) issues
    df_scaled[turnover_column] = scalers['turnover'].transform(df_scaled[turnover_column])

  # Apply scale to categorical features
  df_scaled['hourOfDay'] = df_scaled['hourOfDay'] / 23  # Normalize to [0,1]
  df_scaled['dayOfWeek'] = df_scaled['dayOfWeek'] / 6  # Normalize to [0,1] (0=Monday, 6=Sunday)
  df_scaled['weekOfYear'] = df_scaled['weekOfYear'] / 51  # Normalize to [0,1]
  df_scaled['monthOfYear'] = df_scaled['monthOfYear'] / 11  # Normalize to [0,1] (0=Jan, 11=Dec)
  df_scaled['minuteOfHour'] = df_scaled['minuteOfHour'] / 59  # Normalize to [0,1]

  return df_scaled, scalers

def unscale_features(df, scalers):
  df_unscaled = df.copy()

  # Define columns for scale
  price_columns = ['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'SMA_short', 'SMA_long', 'EMA_short', 'EMA_long']
  price_change_column = ['priceChange', 'priceChange_window_short', 'priceChange_window_long']
  relative_price_change_column = ['relativePriceChange', 'relativePriceChange_window_short', 'relativePriceChange_window_long']
  volume_columns = ['volume']
  turnover_column = ['turnover']
  returns_columns = ['logReturn', 'stdReturn_short', 'stdReturn_long', 'MACD_line', 'MACD_signal', 'MACD_histogram', 
             'Stochastic_K_short', 'Stochastic_D_short', 'Stochastic_K_long', 'Stochastic_D_long', 'ROC_short', 'ROC_long', 'RSI_short', 'RSI_long']
  range_columns = ['highLowRange']

  # Apply different scalers
  df_unscaled[price_columns] = scalers['price'].inverse_transform(df_unscaled[price_columns])
  df_unscaled[price_change_column] = scalers['price_change'].inverse_transform(df_unscaled[price_change_column])
  df_unscaled[relative_price_change_column] = scalers['relative_price_change'].inverse_transform(df_unscaled[relative_price_change_column])
  df_unscaled[volume_columns] = scalers['volume'].inverse_transform(df_unscaled[volume_columns])
  df_unscaled[returns_columns] = scalers['returns'].inverse_transform(df_unscaled[returns_columns])
  df_unscaled[range_columns] = scalers['range'].inverse_transform(df_unscaled[range_columns])

  # Apply log transformation to turnover before scaling
  df_unscaled[turnover_column] = np.log1p(df_unscaled[turnover_column])  # log1p to avoid log(0) issues
  df_unscaled[turnover_column] = scalers['turnover'].inverse_transform(df_unscaled[turnover_column])

  # Apply scale to categorical features
  df_unscaled['hourOfDay'] = df_unscaled['hourOfDay'] * 23  # Normalize to [0,1]
  df_unscaled['dayOfWeek'] = df_unscaled['dayOfWeek'] * 6  # Normalize to [0,1] (0=Monday, 6=Sunday)
  df_unscaled['weekOfYear'] = df_unscaled['weekOfYear'] * 51  # Normalize to [0,1]
  df_unscaled['monthOfYear'] = df_unscaled['monthOfYear'] * 11  # Normalize to [0,1] (0=Jan, 11=Dec)
  df_unscaled['minuteOfHour'] = df_unscaled['minuteOfHour'] * 59  # Normalize to [0,1]

  return df_unscaled

def run_baseline(df, fee_open, fee_close, params):
  realized_pnl = 0

  for i in range(len(df)):
    current_row = df.iloc[i]
    if not np.isnan(current_row['fundingRate']):  # Funding rate
      gross_profit = params['position_amount_open'] * (df['closePrice'].iloc[0] - current_row['closePrice']) / df['closePrice'].iloc[0]
      position_amount_cur = params['position_amount_open'] + gross_profit
      realized_pnl += position_amount_cur * current_row['fundingRate']

  gross_profit = params['position_amount_open'] * (df['closePrice'].iloc[0] - df['closePrice'].iloc[-1]) / df['closePrice'].iloc[0]
  position_amount_cur = params['position_amount_open'] + gross_profit

  realized_pnl -= params['position_amount_open'] * fee_open + position_amount_cur * fee_close
  profit = (gross_profit + realized_pnl) / params['position_amount_open']

  return profit

def run_random_strategy(df, amount, fee_open, fee_close, params):
    position_open = False
    total_amount = amount
    realized_pnl = 0
    profits = []
    positions = []
    position_open_timestamps = []
    position_close_timestamps = []
    position_prices = []
    position_amounts = []
    position_open_timestamp = None

    for i in range(len(df)):
        current_row = df.iloc[i]
        price_close_cur = current_row['closePrice']
        start_time = current_row['startTime']
        funding_rate = current_row['fundingRate']

        if not position_open:
            position_amount_open = min(amount * params['position_amount_open'], total_amount)
            # Randomly decide to open a position (50% chance)
            shouldOpen = random.random() > 0.5 and position_amount_open > 0
            if shouldOpen:
                position_amounts.append(position_amount_open)
                position_prices.append(price_close_cur)
                positions.append(1)
                position_open_timestamps.append(start_time)

                fee_open_amount = position_amount_open * fee_open
                realized_pnl = -fee_open_amount
                total_amount -= position_amount_open + fee_open_amount
                position_open_timestamp = start_time
                position_open = True
        else:
            gross_profit = 0
            position_amount_sum = 0
            for j in range(len(position_amounts)):
                position_amount_sum += position_amounts[j]
                gross_profit += position_amounts[j] * (position_prices[j] - price_close_cur) / position_prices[j]
            position_amount_cur = position_amount_sum + gross_profit

            if not np.isnan(funding_rate):
                funding_rate_amount = position_amount_cur * funding_rate
                realized_pnl += funding_rate_amount
                total_amount += funding_rate_amount

            duration = (start_time - position_open_timestamp).total_seconds() / 3600
            if duration >= 1 or i == len(df) - 1:
                fee_close_amount = position_amount_cur * fee_close
                profit = (gross_profit + realized_pnl - fee_close_amount) / position_amount_sum
                realized_pnl -= fee_close_amount
                total_amount += position_amount_cur - fee_close_amount

                profits.append(profit)
                position_close_timestamps.append(start_time)

                realized_pnl = 0
                position_open_timestamp = None
                position_prices = []
                position_amounts = []
                position_open = False

    return total_amount, profits, positions, position_open_timestamps, position_close_timestamps

def run_bot(df, y, amount, fee_open, fee_close, params):
  position_open = False
  position_open_timestamp = None
  total_amount = amount
  realized_pnl = 0

  profits = []  # Track profit per trade
  positions = []
  position_open_timestamps = []
  position_close_timestamps = []
  position_prices = []
  position_amounts = []

  for i in range(0, len(df)):
    current_row = df.iloc[i]
    price_close_cur = current_row['closePrice']
    start_time = current_row['startTime']
    close_price_diff_1h = current_row['relativePriceChange_window_short']
    funding_rate = current_row['fundingRate']

    price_close_predicted = y[i][0]

    if not position_open:
      position_amount_open = min(amount * params['position_amount_open'], total_amount)
      # Open position
      shouldOpen = \
        close_price_diff_1h > params['price_diff_threshold_open'] \
        and (price_close_cur - price_close_predicted) / price_close_cur > params['price_diff_future_threshold_open'] \
        and position_amount_open > 0
      if shouldOpen:
        position_amounts.append(position_amount_open)
        position_prices.append(price_close_cur)
        positions.append(1)
        position_open_timestamps.append(start_time)

        fee_open_amount = position_amount_open * fee_open
        realized_pnl = -fee_open_amount
        total_amount -= position_amount_open + fee_open_amount
        position_open_timestamp = start_time
        position_open = True
    else:
      gross_profit = 0
      position_amount_sum = 0
      for j in range(len(position_amounts)):
        position_amount_sum += position_amounts[j]
        gross_profit += position_amounts[j] * (position_prices[j] - price_close_cur) / position_prices[j]
      position_amount_cur = position_amount_sum + gross_profit

      # Apply funding rate
      if not np.isnan(funding_rate):
        funding_rate_amount = position_amount_cur * funding_rate
        realized_pnl += funding_rate_amount
        total_amount += funding_rate_amount

      # Close position if necessary
      duration = (start_time - position_open_timestamp).total_seconds() / 3600  # Convert to hours
      fee_close_amount = position_amount_cur * fee_close
      profit = (gross_profit + realized_pnl - fee_close_amount) / position_amount_sum
      shouldClose = \
        (
          (price_close_predicted - price_close_cur) / price_close_cur > params['price_diff_future_threshold_close'] and profit > params['profit_min']
        ) \
        or duration > params['max_duration_hours'] \
        or i == len(df) - 1
      
      if shouldClose:  # Close position if duration exceeds max_duration_hours
        realized_pnl -= fee_close_amount
        total_amount += position_amount_cur - fee_close_amount

        profits.append(profit)
        position_close_timestamps.append(start_time)

        realized_pnl = 0
        position_open_timestamp = None
        position_prices = []
        position_amounts = []
        position_open = False

        continue

      # Increase position if necessary
      closePrice_pct_change_since_open = (price_close_cur - position_prices[-1]) / price_close_cur
      position_amount_increase = min(amount * params['position_amount_increase'], total_amount)
      shouldIncrease = \
        closePrice_pct_change_since_open > params['price_diff_threshold_increase'] \
        and (price_close_cur - price_close_predicted) / price_close_cur > params['price_diff_future_threshold_increase'] \
        and position_amount_increase > 0
      if shouldIncrease:
        position_amounts.append(position_amount_increase)
        position_prices.append(price_close_cur)
        positions[-1] += 1
        position_open_timestamps.append(start_time)

        fee_open_amount = position_amount_increase * fee_open
        realized_pnl -= fee_open_amount
        total_amount -= position_amount_increase + fee_open_amount

        continue

  return total_amount, profits, positions, position_open_timestamps, position_close_timestamps

def run_bot_simple(df, y, amount, fee_open, fee_close, params):
  position_open = False
  position_open_timestamp = None
  total_amount = amount
  realized_pnl = 0

  profits = []  # Track profit per trade
  positions = []
  position_open_timestamps = []
  position_close_timestamps = []
  position_prices = []
  position_amounts = []

  for i in range(0, len(df)):
    current_row = df.iloc[i]
    price_close_cur = current_row['closePrice']
    start_time = current_row['startTime']
    funding_rate = current_row['fundingRate']

    price_close_predicted = y[i][0]

    if not position_open:
      position_amount_open = min(amount * params['position_amount_open'], total_amount)
      # Open position
      shouldOpen = \
        (price_close_cur - price_close_predicted) / price_close_cur > params['price_diff_future_threshold_open'] \
        and position_amount_open > 0
      if shouldOpen:
        position_amounts.append(position_amount_open)
        position_prices.append(price_close_cur)
        positions.append(1)
        position_open_timestamps.append(start_time)

        fee_open_amount = position_amount_open * fee_open
        realized_pnl = -fee_open_amount
        total_amount -= position_amount_open + fee_open_amount
        position_open_timestamp = start_time
        position_open = True
    else:
      gross_profit = 0
      position_amount_sum = 0
      for j in range(len(position_amounts)):
        position_amount_sum += position_amounts[j]
        gross_profit += position_amounts[j] * (position_prices[j] - price_close_cur) / position_prices[j]
      position_amount_cur = position_amount_sum + gross_profit

      # Apply funding rate
      if not np.isnan(funding_rate):
        funding_rate_amount = position_amount_cur * funding_rate
        realized_pnl += funding_rate_amount
        total_amount += funding_rate_amount

      # Close position if necessary
      duration = (start_time - position_open_timestamp).total_seconds() / 3600  # Convert to hours
      fee_close_amount = position_amount_cur * fee_close
      profit = (gross_profit + realized_pnl - fee_close_amount) / position_amount_sum

      # Stop-loss and take-profit
      stop_loss = -0.02  # 2% loss
      take_profit = 0.1  # 4% profit
      duration = (start_time - position_open_timestamp).total_seconds() / 3600

      shouldClose = (
          profit <= stop_loss or
          profit >= take_profit or
          duration >= 1 or
          i == len(df) - 1
      )
      
      if shouldClose:
        realized_pnl -= fee_close_amount
        total_amount += position_amount_cur - fee_close_amount

        profits.append(profit)
        position_close_timestamps.append(start_time)

        realized_pnl = 0
        position_open_timestamp = None
        position_prices = []
        position_amounts = []
        position_open = False

  return total_amount, profits, positions, position_open_timestamps, position_close_timestamps

def run_bot_simple_v2(df, y, amount, fee_open, fee_close, params):
    position_open = False
    position_open_timestamp = None
    total_amount = amount
    realized_pnl = 0

    profits = []
    positions = []
    position_open_timestamps = []
    position_close_timestamps = []
    position_prices = []
    position_amounts = []

    # Calculate ATR and SMA
    df['atr'] = (df['highPrice'] - df['lowPrice']).rolling(window=14).mean()
    df['sma_20'] = df['closePrice'].rolling(window=20).mean()

    for i in range(0, len(df)):
        current_row = df.iloc[i]
        price_close_cur = current_row['closePrice']
        start_time = current_row['startTime']
        funding_rate = current_row['fundingRate']
        atr = current_row['atr']
        sma_20 = current_row['sma_20']
        price_close_predicted = y[i][0]

        if not position_open:
            position_amount_open = min(amount * params['position_amount_open'], total_amount)
            # Volatility-adjusted threshold
            dynamic_threshold = params['price_diff_future_threshold_open'] * (1 + (atr / price_close_cur if not pd.isna(atr) else 0))
            # Trend filter
            is_downtrend = price_close_cur < sma_20 if not pd.isna(sma_20) else True
            shouldOpen = (
                (price_close_cur - price_close_predicted) / price_close_cur > dynamic_threshold
                and position_amount_open > 0
                and is_downtrend
            )
            if shouldOpen:
                position_amounts.append(position_amount_open)
                position_prices.append(price_close_cur)
                positions.append(1)
                position_open_timestamps.append(start_time)

                fee_open_amount = position_amount_open * fee_open
                realized_pnl = -fee_open_amount
                total_amount -= position_amount_open + fee_open_amount
                position_open_timestamp = start_time
                position_open = True
        else:
            gross_profit = 0
            position_amount_sum = 0
            for j in range(len(position_amounts)):
                position_amount_sum += position_amounts[j]
                gross_profit += position_amounts[j] * (position_prices[j] - price_close_cur) / position_prices[j]
            position_amount_cur = position_amount_sum + gross_profit

            if not np.isnan(funding_rate):
                funding_rate_amount = position_amount_cur * funding_rate
                realized_pnl += funding_rate_amount
                total_amount += funding_rate_amount

            current_profit = gross_profit / position_amount_sum
            stop_loss = -0.02  # 2% loss
            take_profit = 0.04  # 4% profit
            duration = (start_time - position_open_timestamp).total_seconds() / 3600

            shouldClose = (
                current_profit <= stop_loss or
                current_profit >= take_profit or
                duration >= 1 or
                i == len(df) - 1
            )

            if shouldClose:
                fee_close_amount = position_amount_cur * fee_close
                profit = (gross_profit + realized_pnl - fee_close_amount) / position_amount_sum
                realized_pnl -= fee_close_amount
                total_amount += position_amount_cur - fee_close_amount

                profits.append(profit)
                position_close_timestamps.append(start_time)

                realized_pnl = 0
                position_open_timestamp = None
                position_prices = []
                position_amounts = []
                position_open = False

    return total_amount, profits, positions, position_open_timestamps, position_close_timestamps
  
def run_bot_v3(df, y, amount, fee_open, fee_close, params, start_idx=0, end_idx=None):
  if end_idx is None:
    end_idx = len(df)

  profits = []  # Track profit per trade
  positions = []

  position_open = False
  position_open_timestamp = None
  total_amount = amount
  realized_pnl = 0

  position_open_timestamps = []
  position_close_timestamps = []
  position_prices = []
  position_amounts = []

  for i in range(start_idx, end_idx):
    current_row = df.iloc[i]
    price_close_cur = current_row['closePrice']
    start_time = current_row['startTime']
    rsi_1h = current_row['RSI_short']
    close_price_diff_1h = current_row['relativePriceChange_window_short']
    funding_rate = current_row['fundingRate']
    price_close_predicted = y[i][0]

    if not position_open :  # Open short
      position_amount_open = min(amount * params['position_amount_open'], total_amount)
      
      shouldOpen = close_price_diff_1h > params['price_diff_threshold_open'] \
        and rsi_1h > params['rsi_threshold_open'] \
        and position_amount_open > 0 \
        and (price_close_cur - price_close_predicted) / price_close_cur > 0.1

      if shouldOpen:
        position_amounts.append(position_amount_open)
        position_prices.append(price_close_cur)
        positions.append(1)
        position_open_timestamps.append(start_time)

        fee_open_amount = position_amount_open * fee_open
        realized_pnl = -fee_open_amount
        total_amount -= position_amount_open + fee_open_amount
        position_open_timestamp = start_time
        position_open = True
    elif position_open:
      gross_profit = 0
      position_amount_sum = 0
      for j in range(len(position_amounts)):
        position_amount_sum += position_amounts[j]
        gross_profit += position_amounts[j] * (position_prices[j] - price_close_cur) / position_prices[j]
      position_amount_cur = position_amount_sum + gross_profit

      if not np.isnan(funding_rate):  # Funding rate
        funding_rate_amount = position_amount_cur * funding_rate
        realized_pnl += funding_rate_amount
        total_amount += funding_rate_amount

      # Close position if position duration over threshold or last record
      duration = (start_time - position_open_timestamp).total_seconds() / 3600  # Convert to hours

      if duration > params['max_duration_hours'] or i == end_idx - 1:  # Close position if duration exceeds max_duration_hours
        fee_close_amount = position_amount_cur * fee_close
        realized_pnl -= fee_close_amount
        total_amount += position_amount_cur - fee_close_amount

        profit = (gross_profit + realized_pnl) / position_amount_sum
        profits.append(profit)
        position_close_timestamps.append(start_time)

        realized_pnl = 0
        position_open_timestamp = None
        position_prices = []
        position_amounts = []
        position_open = False

        continue

      # Close position if rsi below threshold and profit over threshold
      shouldClose = rsi_1h < params['rsi_threshold_close']
      if shouldClose:
        fee_close_amount = position_amount_cur * fee_close
        profit = (gross_profit + realized_pnl - fee_close_amount) / position_amount_sum
        if profit > params['profit_min']:
          realized_pnl -= fee_close_amount
          total_amount += position_amount_cur - fee_close_amount

          profits.append(profit)
          position_close_timestamps.append(start_time)

          realized_pnl = 0
          position_open_timestamp = None
          position_prices = []
          position_amounts = []
          position_open = False

          continue

      # Increase position if necessary
      closePrice_pct_change_since_open = (price_close_cur - position_prices[-1]) / price_close_cur
      position_amount_increase = min(amount * params['position_amount_increase'], total_amount)
      shouldIncrease = closePrice_pct_change_since_open > params['price_diff_threshold_increase'] \
        and rsi_1h > params['rsi_threshold_increase'] \
        and position_amount_increase > 0
      if shouldIncrease:  # Increase short
        position_amounts.append(position_amount_increase)
        position_prices.append(price_close_cur)
        positions[-1] += 1
        position_open_timestamps.append(start_time)

        fee_open_amount = position_amount_increase * fee_open
        realized_pnl -= fee_open_amount
        total_amount -= position_amount_increase + fee_open_amount

        continue

  return total_amount, profits, positions, position_open_timestamps, position_close_timestamps
