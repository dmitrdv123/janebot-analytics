# Данные

Данные берутся с биржи Bybit в разделе market, например:
https://bybit-exchange.github.io/docs/v5/market/kline

Исследованные криптовалютные пары:
BTCUSDT

А также мемтокены
BROCCOLIUSDT
MELANIAUSDT
TRUMPUSDT
FARTCOINUSDT

Основа - это KLine, сгруппированные по временым меткам (1 мин, 5 мин и тд) данные формата:
startTime,openPrice,highPrice,lowPrice,closePrice,volume,turnover

Также используются funding rate при работе бота:
symbol,fundingRate,fundingRateTimestamp

При открытии и закрытии короткой (short, зарабатывает при падении) и длинной позиции (long, зарабатывает при росте) берутся комиссии:
1) комиссия при открытии позиции
2) комиссия при закрытии позиции
3) каждые N часов если позиция открыта, то начисляется плавающая комиссия из funding rate. Если позиция закрыта раньше времени начисления, то она не применяется. Причем если ставка funding rate > 0, позиция короткая, то тебе платят. Если отрицательная, то ты платишь. И все наоборот, если позиция длинная

На основе funding rate можно вычислять также фичи, так как скажем большая положительная funding rate, говорит о том, что открыто много длинных позиций. Из полезных данных для обучения также можно выделить:
order book - книга заявок (все заявки на открытие и закрытие позиций в данный момент времени, их цена и объемы)
long short ratio 
и тд. Их все можно просмотреть по ссылке выше

# Подготовка

python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

# Скачивание данных

Скрипт для скачивания данных - [download_data.py](scripts/download_data.py)

Укажите пару, интервал и период:
symbol = 'BROCCOLIUSDT'
interval = '5'  # Kline interval (1m, 5m, 15m, etc.)
period_long_short_ratio = '5min'  # Period for Long/Short Ratio (5min 15min 30min 1h 4h 4d)
intervalTime = '5min'  # Interval Time for Open Interest (5min 15min 30min 1h 4h 1d)
start_time = datetime(2025, 3, 23)
end_time = datetime(2025, 3, 28)

Раскомментарьте строки, которые будут скачивать необходимые вам данные, например:
```
  download_kline(symbol, interval, start_time, end_time)
  download_funding_rate(symbol, start_time, end_time)
```

Запустите и результат будет в каталоге data.

# Расчет фич

Скрипт для расчета фич - [calc_features.py](scripts/calc_features.py)

Укажите пару, интервал и период:
symbol = 'BROCCOLIUSDT'
interval = '5'  # Kline interval (1m, 5m, 15m, etc.)
intervalTime = '5min'  # Interval Time for Open Interest (5min 15min 30min 1h 4h 1d)
period_long_short_ratio = '5min'  # Period for Long/Short Ratio (5min 15min 30min 1h 4h 4d)

Раскомментарьте строки, которые будут рассчитывать фичи, например:
```
  df_features_kline = calc_features_kline(symbol, interval)
  save_features(df_features_kline, 'features/kline', symbol, interval)
```

Запустите и результат будет в каталоге features.

# Модели

## Модель на основе правил

### Описание

Реализована в скрипте [train_model_broccoli_rule_based.py](scripts/train_model_broccoli_rule_based.py)

Гипотеза в том, что цена мемтокена - это убывающая экспонента в результате спада первоначального хайпа. При этом есть локальный рост в результате каких то внешних новостей, за которым следует падение. Таким образом, основная идея стратегии открывать короткие позиции при росте, если рост продолжается, то увеличивать позицию. И закрывать позицию при получении прибыли.

В стратегии есть следующие настраиваемые параметры:
- position_amount_open - есть общая сумма которой мы располагаем. Этот параметр - это доля этой суммы которую мы используем для открытия позиции. Например, у нас есть 1000 USDT, position_amount_open=0.1 означает, что мы возьмем 100 USDT для открытия новой позиции
- position_amount_increase - аналогично, только это доля используемая для увеличения позиции
- profit_min - доля прибыли при которой мы закрываем позицию
- price_diff_threshold_open - увеличение цены, при которой мы открываем позицию
- price_diff_threshold_increase - увеличение цены, при которой мы увеличиваем позицию
- rsi_threshold_open - значение индикатора RSI за последний час, при котором мы открываем позицию. Высокий RSI может свидетельствовать о прекращении дальнейшего роста.
- rsi_threshold_increase - значение индикатора RSI за последний час, при котором мы увеличиваем позицию
- rsi_threshold_close - значение индикатора RSI за последний час, при котором мы закрываем позицию. Низкий RSI может свидетельствовать о прекращении дальнейшего падения.
- max_duration_hours - максимальное время в течении, которого мы готовы держать позицию

В скрипте есть начальные значения этих параметров. А также реализован генетический алгоритм поиска параметров, при котором будет максимальная прибыль.

### Результаты

Вводные данные:
- symbol = 'BROCCOLIUSDT'
- interval = '5'  # Kline interval (1m, 5m, 15m, etc.)
- amount = 10000
- params = {
    'position_amount_open': 0.1,
    'position_amount_increase': 0.05,
    'profit_min': 0.1,
    'price_diff_threshold_open': 0.1,
    'price_diff_threshold_increase': 0.05,
    'rsi_threshold_open': 80,
    'rsi_threshold_increase': 80,
    'rsi_threshold_close': 20,
    'max_duration_hours': 5 * 24,
  }

Итог:
- Весь временной ряд:
  - Baseline Total Profit/Loss: 12.94 %
  - Total Profit/Loss: *168.49 %*
  - Total Trade Profit/Loss: 187.09 %
  - Avg Profit per Trade: 9.85 %
  - Profit Factor: 4.5594
  - Expected Value (EV): 9.8468 %
  - Reward to Risk Ration (RRR): 121.5827 %
  - Количество позиций: 19
  - История увеличения позиций (каждая цифра - это количество раз когда позиция увеличивалась): [1, 4, 6, 8, 2, 4, 4, 14, 13, 7, 4, 14, 10, 10, 4, 12, 9, 14, 14]
  - Max Position Duration: 1 days 22:30:00
  - params = {
    "position_amount_open": 0.4916273264176523,
    "position_amount_increase": 0.1838246840124815,
    "profit_min": 0.13834646615678947,
    "price_diff_threshold_open": 0.09736287176440007,
    "price_diff_threshold_increase": -0.024186683457065337,
    "rsi_threshold_open": 70,
    "rsi_threshold_increase": 73.77586961691634,
    "rsi_threshold_close": 20.133304323016677,
    "max_duration_hours": 46.478455894134946
  }
- Последние 20% данных:
  - Baseline Total Profit/Loss: -167.54 %
  - Total Profit/Loss: *25.70 %*
  - Total Trade Profit/Loss: 25.70 %
  - Avg Profit per Trade: 25.70 %
  - Profit Factor: 0.2570
  - Expected Value (EV): 25.7020 %
  - Reward to Risk Ration (RRR): inf %
  - Количество позиций: 1
  - История увеличения позиций: [4]
  - Max Position Increasing: 4
  - Position Durations: ['1 days 19:15:00']
  - Max Position Duration: 1 days 19:15:00
  - params = {
    "position_amount_open": 0.3917009962429375,
    "position_amount_increase": 0.20287963293112693,
    "profit_min": 0.3128360345134498,
    "price_diff_threshold_open": 0.44936531115014033,
    "price_diff_threshold_increase": 0.03508761109234881,
    "rsi_threshold_open": 82.07034712789493,
    "rsi_threshold_increase": 78.81086128087686,
    "rsi_threshold_close": 15.197949017744893,
    "max_duration_hours": 43.232383175081445
  }
- Последние 20% данных, но с параметрами заданными вручную на основе найденных на всем объеме данных:
  - Baseline Total Profit/Loss: -167.54 %
  - Total Profit/Loss: *18.58 %*
  - Total Trade Profit/Loss: 16.32 %
  - Avg Profit per Trade: 4.08 %
  - Profit Factor: 2.9611
  - Expected Value (EV): 4.0808 %
  - Positions Count: 4
  - Positions Increasing: [4, 4, 4, 3]
  - Max Position Increasing: 4
  - Position Durations: ['1 days 19:10:00', '0 days 15:30:00', '2 days 00:05:00', '0 days 01:35:00']
  - Max Position Duration: 2 days 00:05:00
  - params = {
    "position_amount_open": 0.5,
    "position_amount_increase": 0.2,
    "profit_min": 0.1,
    "price_diff_threshold_open": 0.1,
    "price_diff_threshold_increase": 0.01,
    "rsi_threshold_open": 70,
    "rsi_threshold_increase": 70,
    "rsi_threshold_close": 20,
    "max_duration_hours": 48
  }

## Модель на машинного обучения

Реализована в скриптах:
- [train_model_broccoli_price_prediction_run.py](scripts/train_model_broccoli_price_prediction_run.py) - использование предсказаний модели для открытия и закрытия позиций

Идея в том, чтобы заменить простые правила из предыдущей модели, такие как изменение цены и RSI, на то, что предсказывала бы мат модель, использующая множество параметров.

### Обучение модели

Реализовано в скрипте [train_model_broccoli_price_prediction.py](scripts/train_model_broccoli_price_prediction.py). Реализованы LSTM, LSTM + attention, bidirectional LSTN. На входе фичи, рассчитанные на основе kline (см. метод calc_features_kline в [calc_features.py](scripts/calc_features.py)). Значение для предсказания - цена closePrice через 1 час. Входные и выходные данные нормируются.

Результаты для LSTM:
- Epoch 59: early stopping
- Restoring model weights from the end of the best epoch: 9.
- Train metrics - Loss (MSE): 0.022770, MAE: 0.080366
- Test metrics - Loss (MSE): 0.046232, MAE: 0.094575
- Train MAPE: 3.81%
- Test MAPE: 3.52%
- Train MSPE: 0.24%
- Test MSPE: 0.42%
- Train Directional Accuracy: 0.4924
- Test Directional Accuracy: 0.4934

Анализ результатов:
- MAPE (3.52%): An average error of 3.52% is too high to consistently generate profits, especially with transaction fees of 0.04% round-trip. The model’s predictions need to be more accurate to capture price movements that exceed the fees and provide a profit margin.
- Directional Accuracy (49.34%): The random directional accuracy is a major issue. Your trading strategy relies on the model predicting whether the price will drop (to open a short position). If the model can’t predict the direction better than a coin flip, the strategy is unlikely to be profitable in the long run.

### Использование модели

TODO: взять rule based скрипт и использовать предсказание модели вместо или совместно для принятия решений об открытии, увеличении, закрытии позиций. Но видимо вначале надо добиться большего угадывания направления движения.
