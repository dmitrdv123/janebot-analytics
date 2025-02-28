import numpy as np
import tensorflow as tf
import joblib

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils_features import calc_features_kline_based
from utils import load_data

def load_model_and_scalers(path='models', model='model_kline_lstm.keras'):
  # Load the trained LSTM model
  model = tf.keras.models.load_model(f'{path}/{model}')
  
  # Load the scalers
  scalers = {
    'target': joblib.load(f'{path}/model_kline_scaler_target.pkl'),
    'price': joblib.load(f'{path}/model_kline_scaler_price.pkl'),
    'price_change': joblib.load(f'{path}/model_kline_scaler_price_change.pkl'),
    'volume': joblib.load(f'{path}/model_kline_scaler_volume.pkl'),
    'turnover': joblib.load(f'{path}/model_kline_scaler_turnover.pkl'),
    'returns': joblib.load(f'{path}/model_kline_scaler_returns.pkl'),
    'range': joblib.load(f'{path}/model_kline_scaler_range.pkl')
  }
  return model, scalers

def load_data_kline(symbol='BTCUSDT', interval=1):
  # Load the Kline Data
  df_data = load_data(f'data/kline/{symbol}/{interval}')

  # Ensure numeric columns are properly cast
  df_data['startTime'] = df_data['startTime'].astype(float)
  df_data['openPrice'] = df_data['openPrice'].astype(float)
  df_data['highPrice'] = df_data['highPrice'].astype(float)
  df_data['lowPrice'] = df_data['lowPrice'].astype(float)
  df_data['closePrice'] = df_data['closePrice'].astype(float)
  df_data['volume'] = df_data['volume'].astype(float)
  df_data['turnover'] = df_data['turnover'].astype(float)

  # Sort by timestamp
  df_data = df_data.sort_values(by='startTime')

  return df_data

def scale_features_kline(df, scalers):
  # Define columns for scale 
  price_columns = ['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10']
  price_change_column = ['priceChange']
  volume_columns = ['volume']
  turnover_column = ['turnover']
  returns_columns = ['logReturn', 'stdReturn_5m', 'stdReturn_10m', 'MACD_line', 'MACD_signal', 'MACD_histogram', 
             'Stochastic_K', 'Stochastic_D', 'ROC_14', 'RSI_14']
  range_columns = ['highLowRange']

  df_scaled = df.copy()
  
  # Apply different scalers
  df_scaled[price_columns] = scalers['price'].transform(df_scaled[price_columns])
  df_scaled[price_change_column] = scalers['price_change'].transform(df_scaled[price_change_column])
  df_scaled[volume_columns] = scalers['volume'].transform(df_scaled[volume_columns])
  df_scaled[returns_columns] = scalers['returns'].transform(df_scaled[returns_columns])
  df_scaled[range_columns] = scalers['range'].transform(df_scaled[range_columns])
  # Apply log transformation to turnover before scaling
  df_scaled[turnover_column] = np.log1p(df_scaled[turnover_column])  # log1p to avoid log(0) issues
  df_scaled[turnover_column] = scalers['turnover'].transform(df_scaled[turnover_column])

  # Apply scale to categorical features
  df_scaled['hourOfDay'] = df_scaled['hourOfDay'] / 23  # Normalize to [0,1]
  df_scaled['dayOfWeek'] = df_scaled['dayOfWeek'] / 6    # Normalize to [0,1] (0=Monday, 6=Sunday)
  df_scaled['weekOfYear'] = df_scaled['weekOfYear'] / 51 # Normalize to [0,1]
  df_scaled['monthOfYear'] = df_scaled['monthOfYear'] / 11 # Normalize to [0,1] (0=Jan, 11=Dec)
  df_scaled['minuteOfHour'] = df_scaled['minuteOfHour'] / 59 # Normalize to [0,1]

  return df_scaled

def predict_direction(predicted_price, current_price, min_change_percent=0.06):
  '''Определяет направление движения: 1 (рост), -1 (падение), 0 (стагнация).'''
  price_change_percent = (predicted_price - current_price) / current_price * 100
  if price_change_percent > min_change_percent:
    return 1  # Long
  elif price_change_percent < -min_change_percent:
    return -1  # Short
  return 0  # Hold

if __name__ == '__main__':
  # Настройки бота
  total_amount = 10000  # общая сумма в USDT
  percent_per_trade = 10  # процент от total_amount для каждой позиции (10%)
  fee_open = 0.0002  # 0.02% комиссия за открытие
  fee_close = 0.0002  # 0.02% комиссия за закрытие
  delta = 0.003  # 0.3% минимальная прибыль сверх комиссий
  loss_threshold = -0.01  # порог убытка для закрытия позиции (-1% от позиции)

  window_size = 16 # размер окна для предсказания цены

  # Минимальное изменение цены для открытия (включая комиссии и delta)
  min_change_percent = (fee_open + fee_close + delta) * 100

  # Load model and scalers
  model, scalers = load_model_and_scalers()

  # Load data
  df_data = load_data_kline(symbol='BTCUSDT', interval=5)

  # Define feature columns
  input_columns = [
    'openPrice', 
    'highPrice', 
    'lowPrice', 
    'closePrice', 
    'volume', 
    'turnover', 
    'priceChange', 
    'relativePriceChange', 
    'logReturn', 
    'SMA_5', 
    'SMA_10', 
    'EMA_5', 
    'EMA_10',
    'hourOfDay', 
    'dayOfWeek', 
    'weekOfYear', 
    'monthOfYear', 
    'minuteOfHour', 
    'isWeekend', 
    'highLowRange', 
    'stdReturn_5m', 
    'stdReturn_10m', 
    'RSI_14', 
    'MACD_line', 
    'MACD_signal', 
    'MACD_histogram', 
    'Stochastic_K', 
    'Stochastic_D', 
    'ROC_14',
  ]

  # Calculate features
  df_features = calc_features_kline_based(df_data)
  
  # Scale features
  df_features_scaled = scale_features_kline(df_features, scalers)
  
  # Drop NaN values (last row will be NaN after shifting)
  df_features_scaled.dropna(inplace=True)
  
  # Предсказание
  X = df_features_scaled[input_columns].values
  X = np.array(X, dtype=np.float32).reshape(X.shape[0], 1, X.shape[1])
  y_pred = model.predict(X, verbose=0).flatten()
  predicted_prices = scalers['target'].inverse_transform(y_pred.reshape(-1, 1)).flatten()
  actual_prices = scalers['target'].inverse_transform(df_features_scaled[['closePrice']]).flatten()
  times = df_features_scaled['startTime'].tolist()

  # Lists to store results
  actual_values = []  # Будет содержать futureClosePrice (следующие цены)
  predicted_values = []
  total_balances = [total_amount]  # История баланса
  profits = []
  signals = []
  unrealized_pnl_history = []  # История нереализованной прибыли/убытков

  position_open = False
  position_type = None  # 'Long' или 'Short'
  position_amount = 0  # Размер позиции в USDT
  entry_price = 0  # Цена входа
  unrealized_pnl = 0  # Нереализованная прибыль/убыток
  long_signals = 0
  short_signals = 0
  long_correct = 0
  short_correct = 0

  # Цикл для тестирования бота
  for i in range(len(df_features_scaled) - 1):  # -1, так как используем futureClosePrice
    predicted_close = predicted_prices[i]
    current_close = actual_prices[i]
    next_actual_close = actual_prices[i + 1]
    current_time = times[i]
    
    # Сохраняем предсказанную цену и следующую актуальную цену
    actual_values.append(next_actual_close)  # Сдвигаем actual_values вперёд
    predicted_values.append(predicted_close)

    # Логика бота
    if not position_open:
      direction = predict_direction(predicted_close, current_close, min_change_percent)

      # Условие для открытия позиции
      if direction == 1 or direction == -1:
        position_amount = total_amount * (percent_per_trade / 100)
        total_amount -= position_amount
        entry_price = current_close
        position_open = True
        position_type = 'Long' if direction == 1 else 'Short'
        unrealized_pnl = -position_amount * fee_open  # Учитываем комиссию за открытие
        signals.append(position_type)
        if position_type == 'Long':
          long_signals += 1
        else:
          short_signals += 1
        print(f'Opened {position_type} position at {entry_price:.2f} with {position_amount:.2f} USDT at time {current_time}, Unrealized PnL: {unrealized_pnl:.2f}')
    else:
      direction = predict_direction(predicted_close, current_close, min_change_percent)
      
      # Расчёт нереализованной прибыли с учётом текущей цены
      if position_type == 'Long':
        gross_profit = (current_close - entry_price) / entry_price * position_amount
        current_position_value = position_amount + gross_profit
        close_fee = current_position_value * fee_close if current_position_value > 0 else 0
        unrealized_pnl = gross_profit - position_amount * fee_open - close_fee

        if unrealized_pnl < 0 or unrealized_pnl > 5 or direction != 1:  # Закрываем, если позиция убыточна или следующее движение против позиции
          if current_close > entry_price:
            long_correct += 1

          total_amount += position_amount + unrealized_pnl
          print(f'Closed Long at {current_close:.2f} due to loss. Unrealized PnL: {unrealized_pnl:.2f}, Total: {total_amount:.2f}')
          position_open = False
          position_amount = 0
          entry_price = 0
          unrealized_pnl = 0
        else:
          print(f'Holding Long at {current_close:.2f}. Unrealized PnL: {unrealized_pnl:.2f}')

      elif position_type == 'Short':
        gross_profit = (entry_price - current_close) / entry_price * position_amount
        current_position_value = position_amount + gross_profit
        close_fee = current_position_value * fee_close if current_position_value > 0 else 0
        unrealized_pnl = gross_profit - position_amount * fee_open - close_fee

        if unrealized_pnl < 0 or unrealized_pnl > 5 or direction != -1:  # Закрываем, если позиция убыточна или следующее движение против позиции
          if current_close < entry_price:
            short_correct += 1
            
          total_amount += position_amount + unrealized_pnl
          print(f'Closed Short at {current_close:.2f} due to loss. Unrealized PnL: {unrealized_pnl:.2f}, Total: {total_amount:.2f}')
          position_open = False
          position_amount = 0
          entry_price = 0
          unrealized_pnl = 0

        else:
          print(f'Holding Short at {current_close:.2f}. Unrealized PnL: {unrealized_pnl:.2f}')

    total_balance = total_amount + position_amount + unrealized_pnl
    total_balances.append(total_balance)
    profits.append(total_balance - total_balances[0])
    unrealized_pnl_history.append(unrealized_pnl)

  if position_open:
    total_amount += position_amount + unrealized_pnl
    print(f'Closed final {position_type} at {current_close:.2f}. Unrealized PnL: {unrealized_pnl:.2f}, Total: {total_amount:.2f}')
    position_open = False
    position_amount = 0
    entry_price = 0
    unrealized_pnl = 0

  # Итоговые метрики
  total_signals = long_signals + short_signals
  total_correct = long_correct + short_correct
  accuracy = total_correct / total_signals * 100 if total_signals > 0 else 0
  avg_profit_per_trade = (total_amount - total_balances[0]) / total_signals if total_signals > 0 else 0

  print(f"\nИтоговые результаты бота:")
  print(f"Всего сигналов: {total_signals}")
  print(f"Сигналов Long: {long_signals}")
  print(f"Точных Long: {long_correct} ({long_correct/long_signals*100:.2f}%)" if long_signals > 0 else f"Точных Long: 0 (0.00%)")
  print(f"Сигналов Short: {short_signals}")
  print(f"Точных Short: {short_correct} ({short_correct/short_signals*100:.2f}%)" if short_signals > 0 else f"Точных Short: 0 (0.00%)")
  print(f"Общая точность сигналов: {accuracy:.2f}%")
  print(f"Общая прибыль/убыток: {total_amount - total_balances[0]:.2f} USD")
  print(f"Средняя прибыль на сделку: {avg_profit_per_trade:.2f} USD")

  # Ошибки предсказания
  mse = mean_squared_error(actual_prices, predicted_prices)
  mae = mean_absolute_error(actual_prices, predicted_prices)
  print(f"\nОшибки предсказания:")
  print(f"MSE: {mse:.6f}")
  print(f"MAE: {mae:.6f}")

  # Графики
  plt.figure(figsize=(14, 12))

  plt.subplot(3, 1, 1)
  plt.plot(times[:-1], actual_values, label="Actual Future Close Price", color="blue", alpha=0.7)  # Сдвинутые цены
  plt.plot(times[:-1], predicted_values, label="Predicted Close Price", color="red", linestyle="dashed", alpha=0.7)
  plt.xlabel("Time")
  plt.ylabel("Close Price (USD)")
  plt.title("Actual vs Predicted Close Price")
  plt.legend()
  plt.grid(True)

  plt.subplot(3, 1, 2)
  plt.plot(times[:-1], total_balances[:-1], label="Total Amount (USDT)", color="green")
  plt.xlabel("Time")
  plt.ylabel("Total Amount (USDT)")
  plt.title("Total Amount Over Time")
  plt.legend()
  plt.grid(True)

  plt.subplot(3, 1, 3)
  plt.plot(times[:-1], unrealized_pnl_history, label="Unrealized PnL (USDT)", color="purple")
  plt.xlabel("Time")
  plt.ylabel("Unrealized PnL (USDT)")
  plt.title("Unrealized Profit and Loss Over Time")
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.show()
