import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ta
from scipy.stats import uniform, randint
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from arch import arch_model

# 1. Data Collection
def get_crypto_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data fetched for {ticker} between {start_date} and {end_date}.")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Load sentiment data to determine date range
sentiment_df = pd.read_csv("bitcoin_sentiments_21_24.csv")
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
end_date = sentiment_df['Date'].max()
start_date = end_date - timedelta(days=1041)  # 3 years (1041 days) before the last sentiment date
start_date = start_date.strftime("%Y-%m-%d")
end_date = end_date.strftime("%Y-%m-%d")

# Fetch price data aligned with sentiment (3 years up to the last sentiment date)
ticker = "BTC-USD"
df = get_crypto_data(ticker, start_date, end_date)
if df is not None and not df.empty:
    df.columns = [col[0] for col in df.columns]  # Flatten MultiIndex
    df.to_csv("bitcoin_historical.csv")
    print(f"Saved historical data ({len(df)} days) to bitcoin_historical.csv")
else:
    print("Failed to fetch historical data.")
    exit(1)

# Merge with sentiment, using inner join for alignment and interpolating missing values
df = df.reset_index().merge(sentiment_df[['Date', 'Accurate Sentiments']], on='Date', how='inner').set_index('Date')
df['Sentiment'] = df['Accurate Sentiments'].interpolate(method='linear', limit_direction='both').fillna(0.5)
df.drop(columns=['Accurate Sentiments'], inplace=True)

# Verify alignment
mismatched_dates = df.index.difference(sentiment_df['Date'])
if len(mismatched_dates) > 0:
    print(f"Warning: {len(mismatched_dates)} dates in price data not found in sentiment data.")

# 2. Feature Engineering
if 'Close' in df.columns:
    df['SMA_5'] = ta.trend.sma_indicator(df['Close'].squeeze(), window=5)
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'].squeeze(), window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'].squeeze(), window=14)
    df['MACD'] = ta.trend.macd(df['Close'].squeeze())
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'].squeeze())
    df['Volatility'] = ta.volatility.average_true_range(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze(), window=30)
    df['BB_upper'], df['BB_lower'] = ta.volatility.bollinger_hband(df['Close'], window=20), ta.volatility.bollinger_lband(df['Close'], window=20)

    # Add short-term features for volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Lagged_Close_1'] = df['Close'].shift(1)
    df['Lagged_Close_3'] = df['Close'].shift(3)

    # Volume-Price Movement Relationship
    df['Price_Change_Abs'] = df['Close'].diff().abs()
    df['Volume_Normalized'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()
    df['Volume_Price_Interaction'] = df['Volume_Normalized'] * df['Price_Change_Abs']

    df.bfill(inplace=True)
else:
    raise KeyError("Error: 'Close' column not found in the data.")

returns_3y = df['Close'].pct_change().dropna() * 100
garch_3y_t = arch_model(returns_3y, vol='Garch', p=1, q=1, dist='t', mean='constant', rescale=True)
fit_3y_t = garch_3y_t.fit(disp='off')
print("Historical Student-t params:", fit_3y_t.params)

# 3. Data Preprocessing
df.dropna(inplace=True)

# Updated features list with MACD_Signal and volume-related features
X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Volatility', 'BB_upper', 'BB_lower', 
        'Sentiment', 'Daily_Return', 'Lagged_Close_1', 'Lagged_Close_3', 'Volume_Normalized', 'Volume_Price_Interaction']]
y = df['Close'].shift(-1)

X = X[:-1]
y = y[:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model without sentiment
X_train_no_sentiment = X_train.drop(columns=['Sentiment'])
X_test_no_sentiment = X_test.drop(columns=['Sentiment'])
scaler_no_sentiment = MinMaxScaler()
X_train_no_sentiment_scaled = scaler_no_sentiment.fit_transform(X_train_no_sentiment)
X_test_no_sentiment_scaled = scaler_no_sentiment.transform(X_test_no_sentiment)

# 4. Model Training
param_distributions = {
    'n_estimators': randint(50, 200),
    'learning_rate': uniform(0.01, 0.19),
    'max_depth': randint(3, 7),
    'gamma': uniform(0, 0.5),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

# Train model without sentiment
model_no_sentiment = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
random_search_no_sentiment = RandomizedSearchCV(
    estimator=model_no_sentiment,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=7,
    verbose=1,
    n_jobs=-1,
    error_score='raise'
)
random_search_no_sentiment.fit(X_train_no_sentiment_scaled, y_train)
best_model_no_sentiment = random_search_no_sentiment.best_estimator_

# Train model with sentiment
model_with_sentiment = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
random_search_with_sentiment = RandomizedSearchCV(
    estimator=model_with_sentiment,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=7,
    verbose=1,
    n_jobs=-1,
    error_score='raise'
)
random_search_with_sentiment.fit(X_train_scaled, y_train)
best_model_with_sentiment = random_search_with_sentiment.best_estimator_

# 5. Model Evaluation
y_pred_no_sentiment = best_model_no_sentiment.predict(X_test_no_sentiment_scaled)
y_pred_with_sentiment = best_model_with_sentiment.predict(X_test_scaled)

def calculate_metrics(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        return np.nan
    mape_val = mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) * 100
    print(f"\n{label}:")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape_val:.2f}%")
    print(f"R-squared: {r2:.2f}%")
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape_val, 'R-squared': r2}

metrics_no_sentiment = calculate_metrics(y_test, y_pred_no_sentiment, "Model Without Sentiment")
metrics_with_sentiment = calculate_metrics(y_test, y_pred_with_sentiment, "Model With Sentiment")

metrics_comparison_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE (%)', 'R-squared (%)'],
    'Without Sentiment': [metrics_no_sentiment[k] for k in ['MSE', 'RMSE', 'MAE', 'MAPE', 'R-squared']],
    'With Sentiment': [metrics_with_sentiment[k] for k in ['MSE', 'RMSE', 'MAE', 'MAPE', 'R-squared']]
})
metrics_comparison_df.to_csv("sentiment_comparison_metrics.csv", index=False)
print("\nSaved comparison metrics to sentiment_comparison_metrics.csv")

# Feature Importance (No Sentiment)
feature_names_no_sentiment = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Volatility', 'BB_upper', 'BB_lower', 
                             'Daily_Return', 'Lagged_Close_1', 'Lagged_Close_3', 'Volume_Normalized', 'Volume_Price_Interaction']
feature_importance_no_sentiment = best_model_no_sentiment.feature_importances_
importance_df_no_sentiment = pd.DataFrame({'Feature': feature_names_no_sentiment, 'Importance': feature_importance_no_sentiment})
importance_df_no_sentiment = importance_df_no_sentiment.sort_values(by='Importance', ascending=False)
print("\nFeature Importance (No Sentiment):")
print(importance_df_no_sentiment)

plt.figure(figsize=(12, 8))
plt.barh(importance_df_no_sentiment['Feature'], importance_df_no_sentiment['Importance'])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance (No Sentiment)")
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("XGBoost_Feature_Importance_No_Sentiment.png")
plt.close()

# Feature Importance (With Sentiment)
feature_names_with_sentiment = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Volatility', 'BB_upper', 'BB_lower', 
                               'Sentiment', 'Daily_Return', 'Lagged_Close_1', 'Lagged_Close_3', 'Volume_Normalized', 'Volume_Price_Interaction']
feature_importance_with_sentiment = best_model_with_sentiment.feature_importances_
importance_df_with_sentiment = pd.DataFrame({'Feature': feature_names_with_sentiment, 'Importance': feature_importance_with_sentiment})
importance_df_with_sentiment = importance_df_with_sentiment.sort_values(by='Importance', ascending=False)
print("\nFeature Importance (With Sentiment):")
print(importance_df_with_sentiment)

plt.figure(figsize=(12, 8))
plt.barh(importance_df_with_sentiment['Feature'], importance_df_with_sentiment['Importance'])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance (With Sentiment)")
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("XGBoost_Feature_Importance_With_Sentiment.png")
plt.close()

# 6. Combined Historical and Future Prediction Workflow
# Prediction Function
def predict_price(last_days_data, model, scaler, include_sentiment=True):
    try:
        if last_days_data.isna().any().any():
            last_days_data = last_days_data.ffill()
        columns = X.columns if include_sentiment else X.columns.drop('Sentiment')
        last_days_data = last_days_data[columns]
        last_data_scaled = scaler.transform(last_days_data)
        predicted_price = model.predict(last_data_scaled)
        return predicted_price
    except ValueError as e:
        print(f"Error during prediction: {e}")
        return None

# Fetch Historical Data (Last 180 Days Before Last Sentiment Date)
def get_historical_data():
    try:
        last_sentiment_date = pd.to_datetime("2024-09-12")  # Hardcoding as per requirement
        realtime_end = last_sentiment_date.strftime("%Y-%m-%d")
        realtime_start = (last_sentiment_date - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
        
        last_price_data = yf.download("BTC-USD", start=realtime_start, end=realtime_end)
        if last_price_data.empty:
            raise ValueError(f"No data fetched for BTC-USD between {realtime_start} and {realtime_end}.")
        
        last_price_data.columns = [col[0] for col in last_price_data.columns]
        
        # Add technical indicators
        last_price_data['SMA_5'] = ta.trend.sma_indicator(pd.Series(last_price_data['Close'].squeeze()), window=5)
        last_price_data['SMA_20'] = ta.trend.sma_indicator(pd.Series(last_price_data['Close'].squeeze()), window=20)
        last_price_data['SMA_50'] = ta.trend.sma_indicator(pd.Series(last_price_data['Close'].squeeze()), window=50)
        last_price_data['RSI'] = ta.momentum.rsi(pd.Series(last_price_data['Close'].squeeze()), window=14)
        last_price_data['MACD'] = ta.trend.macd(pd.Series(last_price_data['Close'].squeeze()))
        last_price_data['MACD_Signal'] = ta.trend.macd_signal(pd.Series(last_price_data['Close'].squeeze()))
        last_price_data['Volatility'] = ta.volatility.average_true_range(
            high=pd.Series(last_price_data['High'].squeeze()),
            low=pd.Series(last_price_data['Low'].squeeze()),
            close=pd.Series(last_price_data['Close'].squeeze()),
            window=30
        )
        last_price_data['BB_upper'], last_price_data['BB_lower'] = ta.volatility.bollinger_hband(last_price_data['Close'], window=20), ta.volatility.bollinger_lband(last_price_data['Close'], window=20)

        # Add new features
        last_price_data['Daily_Return'] = last_price_data['Close'].pct_change()
        last_price_data['Lagged_Close_1'] = last_price_data['Close'].shift(1)
        last_price_data['Lagged_Close_3'] = last_price_data['Close'].shift(3)
        last_price_data['Price_Change_Abs'] = last_price_data['Close'].diff().abs()
        last_price_data['Volume_Normalized'] = (last_price_data['Volume'] - df['Volume'].mean()) / df['Volume'].std()
        last_price_data['Volume_Price_Interaction'] = last_price_data['Volume_Normalized'] * last_price_data['Price_Change_Abs']
        
        # Merge with sentiment data
        sentiment_subset = sentiment_df[(sentiment_df['Date'] >= pd.to_datetime(realtime_start)) & 
                                       (sentiment_df['Date'] <= pd.to_datetime(realtime_end))]
        last_price_data = last_price_data.reset_index().merge(sentiment_subset[['Date', 'Accurate Sentiments']], 
                                                              on='Date', how='left')
        last_price_data['Sentiment'] = last_price_data['Accurate Sentiments'].interpolate(method='linear', limit_direction='both').fillna(0.5)
        last_price_data.drop(columns=['Accurate Sentiments'], inplace=True)
        last_price_data.set_index('Date', inplace=True)
        
        return last_price_data.ffill()
    except Exception as e:
        print(f"Error getting historical data: {e}")
        return None

# Get historical data for the last 180 days
last_180_days_data = get_historical_data()

if last_180_days_data is not None and not last_180_days_data.empty:
    # Predict historical prices (last 180 days)
    actual_prices_180d = last_180_days_data['Close']
    X_180d = last_180_days_data[-len(actual_prices_180d):]
    X_180d_with = X_180d[X.columns]
    X_180d_no = X_180d[X.columns.drop('Sentiment')]
    pred_180d_no = best_model_no_sentiment.predict(scaler_no_sentiment.transform(X_180d_no))
    pred_180d_with = best_model_with_sentiment.predict(scaler.transform(X_180d_with))

    # Future Price Prediction (90 days starting from the last sentiment date)
    last_180_days = df[-180:].copy()
    rolling_window = last_180_days.copy()
    future_days = 90
    future_dates = pd.date_range(start=pd.to_datetime(end_date) + pd.Timedelta(days=1), periods=future_days)

    # Lists to store future predictions
    future_prices_with_sentiment = []
    future_prices_no_sentiment = []
    future_highs_with = []
    future_lows_with = []
    future_highs_no = []
    future_lows_no = []
    future_volumes = []
    future_sentiments = []

    # GARCH Volatility Modeling
    returns = rolling_window['Close'].pct_change().dropna() * 100
    print("180-day Returns Stats (Historical):")
    print(returns.describe())
    print("Kurtosis:", returns.kurtosis())

    garch_model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')
    garch_fit = garch_model.fit(disp='off')
    garch_forecast = garch_fit.forecast(horizon=future_days)
    volatility_forecast = np.sqrt(garch_forecast.variance.values[-1, :]) / 100
    print("Normal volatility (first 5 days):", volatility_forecast[:5])

    garch_model_t = arch_model(returns, vol='Garch', p=1, q=1, dist='t', mean='constant', rescale=True)
    garch_fit_t = garch_model_t.fit(disp='off')
    garch_forecast_t = garch_fit_t.forecast(horizon=future_days)
    volatility_forecast_t = np.sqrt(garch_forecast_t.variance.values[-1, :]) / 100
    print("Student-t volatility (first 5 days):", volatility_forecast_t[:5])
    print("Estimated df:", garch_fit_t.params['nu'] if 'nu' in garch_fit_t.params else "Not fitted")

    garch_forecast_3y = fit_3y_t.forecast(horizon=future_days)
    volatility_forecast_3y = np.sqrt(garch_forecast_3y.variance.values[-1, :]) / 100
    print("Historical Student-t volatility (first 5 days):", volatility_forecast_3y[:5])

    # Robust drift rate calculation
    ema_returns = returns.ewm(span=20, adjust=True).mean()
    recent_drift = ema_returns.iloc[-1] / 100
    recent_volatility = returns.rolling(window=20, min_periods=10).std().iloc[-1] / 100
    drift_rate = recent_drift - 0.5 * recent_volatility**2

    recent_volume_avg = rolling_window['Volume'].iloc[-10:].mean()
    last_sentiment = sentiment_df['Accurate Sentiments'].iloc[-1]

    # Future prediction loop
    for i in range(future_days):
        # Update technical indicators
        rolling_window['SMA_5'] = ta.trend.sma_indicator(pd.Series(rolling_window['Close'].squeeze()), window=5)
        rolling_window['SMA_20'] = ta.trend.sma_indicator(pd.Series(rolling_window['Close'].squeeze()), window=20)
        rolling_window['SMA_50'] = ta.trend.sma_indicator(pd.Series(rolling_window['Close'].squeeze()), window=50)
        rolling_window['RSI'] = ta.momentum.rsi(pd.Series(rolling_window['Close'].squeeze()), window=14)
        rolling_window['MACD'] = ta.trend.macd(pd.Series(rolling_window['Close'].squeeze()))
        rolling_window['MACD_Signal'] = ta.trend.macd_signal(pd.Series(rolling_window['Close'].squeeze()))
        rolling_window['Volatility'] = ta.volatility.average_true_range(
            high=pd.Series(rolling_window['High'].squeeze()),
            low=pd.Series(rolling_window['Low'].squeeze()),
            close=pd.Series(rolling_window['Close'].squeeze()),
            window=30
        )
        rolling_window['BB_upper'], rolling_window['BB_lower'] = ta.volatility.bollinger_hband(rolling_window['Close'], window=20), ta.volatility.bollinger_lband(rolling_window['Close'], window=20)

        # Update new features
        rolling_window['Daily_Return'] = rolling_window['Close'].pct_change()
        rolling_window['Lagged_Close_1'] = rolling_window['Close'].shift(1)
        rolling_window['Lagged_Close_3'] = rolling_window['Close'].shift(3)
        rolling_window['Price_Change_Abs'] = rolling_window['Close'].diff().abs()
        rolling_window['Volume_Normalized'] = (rolling_window['Volume'] - df['Volume'].mean()) / df['Volume'].std()
        rolling_window['Volume_Price_Interaction'] = rolling_window['Volume_Normalized'] * rolling_window['Price_Change_Abs']

        # Predict with sentiment
        temp_df_with = rolling_window[-1:][X.columns]
        next_price_with = predict_price(temp_df_with, best_model_with_sentiment, scaler, include_sentiment=True)

        # Predict without sentiment
        temp_df_no = rolling_window[-1:][X.columns.drop('Sentiment')]
        next_price_no = predict_price(temp_df_no, best_model_no_sentiment, scaler_no_sentiment, include_sentiment=False)

        if next_price_with is not None and next_price_no is not None:
            weight = i / future_days
            chosen_volatility = volatility_forecast_3y[i]

            # Predictions with sentiment
            predicted_close_with = next_price_with[0] * (1 + drift_rate * i / future_days)
            garch_vol_with = chosen_volatility * predicted_close_with
            high_variation_with = garch_vol_with * np.random.uniform(0.6, 4.0)
            low_variation_with = garch_vol_with * np.random.uniform(0.6, 4.0)
            predicted_high_with = predicted_close_with + high_variation_with
            predicted_low_with = predicted_close_with - low_variation_with

            # Predictions without sentiment
            predicted_close_no = next_price_no[0] * (1 + drift_rate * i / future_days)
            garch_vol_no = chosen_volatility * predicted_close_no
            high_variation_no = garch_vol_no * np.random.uniform(0.6, 4.0)
            low_variation_no = garch_vol_no * np.random.uniform(0.6, 4.0)
            predicted_high_no = predicted_close_no + high_variation_no
            predicted_low_no = predicted_close_no - low_variation_no

            # Volume and Sentiment simulation
            volume_variation = np.random.uniform(0.9, 2.0)
            predicted_volume = recent_volume_avg * volume_variation * (1 + drift_rate * i / future_days)
            predicted_sentiment = last_sentiment + np.random.uniform(-1, 1)
            predicted_sentiment = np.clip(predicted_sentiment, -1, 1)

            # Append predictions
            future_prices_with_sentiment.append(predicted_close_with)
            future_prices_no_sentiment.append(predicted_close_no)
            future_highs_with.append(predicted_high_with)
            future_lows_with.append(predicted_low_with)
            future_highs_no.append(predicted_high_no)
            future_lows_no.append(predicted_low_no)
            future_volumes.append(predicted_volume)
            future_sentiments.append(predicted_sentiment)

            # Update rolling window
            new_row = pd.DataFrame({
                'Open': [rolling_window['Close'].iloc[-1]],
                'High': [predicted_high_with],
                'Low': [predicted_low_with],
                'Close': [predicted_close_with],
                'Volume': [predicted_volume],
                'Sentiment': [predicted_sentiment]
            }, index=[rolling_window.index[-1] + pd.Timedelta(days=1)])
            rolling_window = pd.concat([rolling_window, new_row])
            rolling_window = rolling_window[-180:]

            recent_volume_avg = rolling_window['Volume'].iloc[-10:].mean()
            last_sentiment = predicted_sentiment
        else:
            print("Prediction failed. Stopping future forecast.")
            break

    # Save predictions
    predictions_df_with = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close': future_prices_with_sentiment,
        'High': future_highs_with,
        'Low': future_lows_with,
        'Volume': future_volumes,
        'Sentiment': future_sentiments
    })
    predictions_df_with.to_csv("bitcoin_predictions_90d_with_sentiment.csv", index=False)
    print("Saved predictions with sentiment to bitcoin_predictions_90d_with_sentiment.csv")

    predictions_df_no = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close': future_prices_no_sentiment,
        'High': future_highs_no,
        'Low': future_lows_no,
        'Volume': future_volumes
    })
    predictions_df_no.to_csv("bitcoin_predictions_90d_no_sentiment.csv", index=False)
    print("Saved predictions without sentiment to bitcoin_predictions_90d_no_sentiment.csv")

    # Combined Plot: Historical (Actual + Predicted) and Future Predictions
    plt.figure(figsize=(14, 7))
    
    # Historical Data (Last 180 Days)
    plt.plot(actual_prices_180d.index, actual_prices_180d.values, label="Actual Price (Last 180 Days)", color="blue")
    plt.plot(X_180d.index, pred_180d_no, label="Predicted (No Sentiment, Historical)", color="red", linestyle='--')
    plt.plot(X_180d.index, pred_180d_with, label="Predicted (With Sentiment, Historical)", color="green", linestyle='--')
    
    # Future Predictions (Next 90 Days)
    plt.plot(future_dates, future_prices_with_sentiment, label="Predicted (With Sentiment, Future)", color="green", marker='o')
    plt.plot(future_dates, future_prices_no_sentiment, label="Predicted (No Sentiment, Future)", color="red", marker='o')
    plt.fill_between(future_dates, future_lows_with, future_highs_with, color='green', alpha=0.2, label='High-Low Range (With Sentiment)')
    plt.fill_between(future_dates, future_lows_no, future_highs_no, color='red', alpha=0.2, label='High-Low Range (No Sentiment)')
    
    # Plot customization
    plt.title("Bitcoin Price: Actual (Last 180 Days) and Predicted (Next 90 Days)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("combined_historical_and_future_prediction.png")
    plt.close()

    # Print future predictions
    print("\nFuture Bitcoin Price Predictions (All 90 Days):")
    print("{:<12} {:<20} {:<20} {:<20} {:<20} {:<20} {:<15}".format(
        "Date", "Close (With Sentiment)", "High (With)", "Low (With)", "Close (No Sentiment)", "Volume", "Sentiment"))
    print("-" * 130)
    for date, close_w, high_w, low_w, close_n, volume, sentiment in zip(
        future_dates, future_prices_with_sentiment, future_highs_with, future_lows_with, future_prices_no_sentiment, future_volumes, future_sentiments):
        print("{:<12} ${:<19.2f} ${:<19.2f} ${:<19.2f} ${:<19.2f} {:<20.0f} {:<15.2f}".format(
            date.strftime("%Y-%m-%d"), close_w, high_w, low_w, close_n, volume, sentiment))
else:
    print("Insufficient data for prediction and plotting.")