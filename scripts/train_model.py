import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

kline_df = pd.read_csv('features/kline/1m/features.csv')

# Assuming the data has been loaded and processed into 'kline_df'
# Extract features and target
features = kline_df[['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume', 'turnover',
                     'priceChange', 'logReturn', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 
                     'hourOfDay', 'dayOfWeek', 'weekOfYear', 'monthOfYear', 'minuteOfHour', 
                     'isWeekend', 'highLowRange', 'stdReturn_5m', 'stdReturn_10m', 
                     'RSI_14', 'MACD_line', 'MACD_signal', 'MACD_histogram', 
                     'Stochastic_K', 'Stochastic_D', 'ROC_14']]

target = kline_df['closePrice'].shift(-1)  # Shift by 1 to predict the next closePrice
target = target.dropna()  # Drop last row because it has NaN value after shift
features = features.iloc[:-1]  # Drop the last row from features to match target size

# Normalize the features and target
scaler_features = StandardScaler()
scaler_target = StandardScaler()

# Scale the features
features_scaled = scaler_features.fit_transform(features)

# Scale the target (closePrice)
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
# model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

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
model.save('models/kline_model.keras')
print('Model has been saved.')

# Save the feature and target scalers
joblib.dump(scaler_features, 'models/kline_model_scaler_features.pkl')
joblib.dump(scaler_target, 'models/kline_model_scaler_target.pkl')
print('Scalers have been saved.')
