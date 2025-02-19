import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Kline Features
df_kline = pd.read_csv('features/kline/1m/features.csv')

# Convert startTime to datetime
df_kline['startTime'] = pd.to_datetime(df_kline['startTime'], unit='ms')

# Load Order Book Features
df_orderbook = pd.read_csv('features/orderbook/features.csv')

# Convert minute column to datetime
df_orderbook['timestamp'] = pd.to_datetime(df_orderbook['timestamp'], unit='ms')

# Perform an INNER JOIN to keep only timestamps that exist in both datasets
df_merged = pd.merge(df_kline, df_orderbook, left_on='startTime', right_on='timestamp', how='inner')

# Drop redundant timestamp columns
df_merged.drop(columns=['startTime', 'timestamp'], inplace=True)

# Define Kline Features
features_kline = df_merged[['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover',
                             'priceChange', 'logReturn', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10',
                             'hourOfDay', 'dayOfWeek', 'weekOfYear', 'monthOfYear', 'minuteOfHour',
                             'isWeekend', 'highLowRange', 'stdReturn_5m', 'stdReturn_10m',
                             'RSI_14', 'MACD_line', 'MACD_signal', 'MACD_histogram', 
                             'Stochastic_K', 'Stochastic_D', 'ROC_14']]

# Define Order Book Features
features_orderbook = df_merged[['mid_price_mean', 'mid_price_std', 'mid_price_min', 'mid_price_max', 'mid_price_last', 
                                'spread_mean', 'spread_std', 'spread_max', 'relative_spread_mean', 'relative_spread_std', 
                                'total_best_ask_volume_mean', 'total_best_ask_volume_std', 'total_best_ask_volume_sum', 'total_best_ask_volume_max', 
                                'total_best_bid_volume_mean', 'total_best_bid_volume_std', 'total_best_bid_volume_sum', 'total_best_bid_volume_max', 
                                'market_depth_ask_mean', 'market_depth_ask_std', 'market_depth_bid_mean', 'market_depth_bid_std',
                                'order_book_imbalance_mean', 'order_book_imbalance_std',
                                'vwap_ask_mean', 'vwap_bid_mean', 'vwap_total_mean',
                                'volume_imbalance_ratio_mean', 'volume_imbalance_ratio_std',
                                'cumulative_delta_volume_last', 'liquidity_pressure_ratio_mean', 'liquidity_pressure_ratio_std', 
                                'mean_ask_size_mean', 'mean_ask_size_std', 'mean_bid_size_mean', 'mean_bid_size_std',
                                'std_ask_size_mean', 'std_bid_size_mean']]

# Define Target (Next Close Price)
target = df_merged['closePrice'].shift(-1)  # Shift by 1 to predict the next closePrice
target = target.dropna()  # Drop last row because it has NaN value after shift
features_kline = features_kline.iloc[:-1]  # Drop the last row from features to match target size
features_orderbook = features_orderbook.iloc[:-1]  # Drop the last row from features to match target size

# Handle NaN and infinite values: Replace them with a specific value (e.g., 0 or mean)
features_orderbook.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN
features_orderbook.fillna(features_orderbook.mean(), inplace=True)  # Replace NaN with mean

# Normalize Kline features separately
scaler_kline = StandardScaler()
features_kline_scaled = scaler_kline.fit_transform(features_kline)

# Normalize Order Book features separately
scaler_orderbook = StandardScaler()
features_orderbook_scaled = scaler_orderbook.fit_transform(features_orderbook)

# Combine the scaled Kline and Order Book features
features_scaled = pd.concat([pd.DataFrame(features_kline_scaled), pd.DataFrame(features_orderbook_scaled)], axis=1)

# Normalize the target (closePrice)
scaler_target = StandardScaler()
target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dense(1)
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Make predictions on the test set
y_pred_scaled = model.predict(X_test)

# Inverse transform the predictions and the actual values
y_pred = scaler_target.inverse_transform(y_pred_scaled)
y_test_actual = scaler_target.inverse_transform(y_test)

# Calculate and print Mean Squared Error (MSE) and Mean Absolute Error (MAE)
mse = mean_squared_error(y_test_actual, y_pred)
mae = mean_absolute_error(y_test_actual, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')

# Save the model and the scaler
model.save('models/kline_orderbook_model.keras')
print('Model has been saved.')

# Save the feature and target scalers
joblib.dump(scaler_kline, 'models/kline_orderbook_model_scaler_kline.pkl')
joblib.dump(scaler_orderbook, 'models/kline_orderbook_model_scaler_orderbook.pkl')
joblib.dump(scaler_target, 'models/kline_orderbook_model_scaler_target.pkl')
print('Scalers have been saved.')
