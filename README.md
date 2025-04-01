# Bitcoin-price-prediction-with-Sentiment-Analysis-using-XBGoost

This code is a comprehensive script for predicting Bitcoin (BTC) prices using historical price data, sentiment analysis, technical indicators, and machine learning models (XGBoost) combined with volatility modeling (GARCH). It performs data collection, feature engineering, model training, evaluation, and generates future price predictions. Below is a detailed explanation of each section:

---

### **1. Data Collection**
#### **Purpose**: Fetch historical Bitcoin price data and merge it with sentiment data.
- **`get_crypto_data(ticker, start_date, end_date)`**: Downloads price data (e.g., Open, High, Low, Close, Volume) for a given cryptocurrency (BTC-USD) using the `yfinance` library from Yahoo Finance.
- **Sentiment Data**: Loads a CSV file (`bitcoin_sentiments_21_24.csv`) containing sentiment scores aligned with dates. The date range for price data is calculated as 3 years (1041 days) prior to the last sentiment date.
- **Merging**: Price data is merged with sentiment data using an inner join on the `Date` column. Missing sentiment values are interpolated linearly and default to 0.5 if unfillable.
- **Output**: The merged dataset is saved as `bitcoin_historical.csv`.

---

### **2. Feature Engineering**
#### **Purpose**: Create additional features to enhance the predictive power of the model.
- **Technical Indicators**: Using the `ta` (technical analysis) library:
  - **SMA (Simple Moving Average)**: 5-day, 20-day, and 50-day averages of the closing price.
  - **RSI (Relative Strength Index)**: Measures momentum over 14 days.
  - **MACD (Moving Average Convergence Divergence)**: Includes MACD line and signal line.
  - **Volatility**: Average True Range (ATR) over 30 days.
  - **Bollinger Bands**: Upper and lower bands based on a 20-day window.
- **Additional Features**:
  - **Daily_Return**: Percentage change in closing price.
  - **Lagged_Close_1, Lagged_Close_3**: Closing prices shifted by 1 and 3 days.
  - **Volume-Related**: Normalized volume and its interaction with absolute price change.
- **GARCH Model**: Fits a GARCH(1,1) model with Student-t distribution to historical returns (3 years) to estimate volatility parameters.
- **Cleaning**: Missing values are backfilled (`bfill`).

---

### **3. Data Preprocessing**
#### **Purpose**: Prepare data for machine learning.
- **Target Variable (`y`)**: Next day’s closing price (`Close.shift(-1)`).
- **Features (`X`)**: Includes price data, technical indicators, sentiment, and engineered features.
- **Train-Test Split**: Splits data into 80% training and 20% testing sets.
- **Scaling**: Uses `MinMaxScaler` to normalize features to a [0, 1] range.
- **No-Sentiment Variant**: A separate feature set excluding sentiment is created for comparison.

---

### **4. Model Training**
#### **Purpose**: Train XGBoost regression models to predict Bitcoin prices.
- **Hyperparameter Tuning**: Uses `RandomizedSearchCV` to optimize XGBoost parameters (e.g., `n_estimators`, `learning_rate`, `max_depth`) over 50 iterations with 7-fold cross-validation.
- **Two Models**:
  - **Without Sentiment**: Trained on features excluding sentiment.
  - **With Sentiment**: Trained on all features, including sentiment.
- **Objective**: Minimize squared error (`reg:squarederror`).

---

### **5. Model Evaluation**
#### **Purpose**: Assess model performance.
- **Predictions**: Generate predictions for the test set using both models.
- **Metrics**: 
  - **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values.
  - **RMSE (Root Mean Squared Error)**: Square root of MSE, in the same unit as the target.
  - **MAE (Mean Absolute Error)**: Average absolute difference.
  - **MAPE (Mean Absolute Percentage Error)**: Percentage error, excluding zero values.
  - **R-squared**: Percentage of variance explained (0-100%).
- **Output**: Metrics are printed and saved to `sentiment_comparison_metrics.csv`.
- **Feature Importance**: 
  - Plots and saves bar charts of feature importance for both models (`XGBoost_Feature_Importance_No_Sentiment.png` and `XGBoost_Feature_Importance_With_Sentiment.png`).

---

### **6. Combined Historical and Future Prediction Workflow**
#### **Purpose**: Predict historical prices (last 180 days) and forecast future prices (next 90 days).
- **`predict_price()`**: Predicts the next price using a trained model and scaled input data.
- **`get_historical_data()`**: Fetches the last 180 days of price data (up to 2024-09-12), adds technical indicators, and merges with sentiment data.
- **Historical Prediction**:
  - Uses both models to predict prices for the last 180 days and compares them to actual prices.
- **Future Prediction**:
  - **Setup**: Starts with the last 180 days of historical data and iterates for 90 future days.
  - **Volatility**: Uses GARCH(1,1) models (Normal and Student-t distributions) to forecast volatility over 90 days based on historical returns.
  - **Drift**: Calculates a drift rate using an exponential moving average of returns and recent volatility.
  - **Simulation**:
    - Predicts closing prices using both models.
    - Estimates high/low prices using GARCH volatility with random variation.
    - Simulates volume and sentiment with random noise.
  - **Rolling Window**: Updates features dynamically for each future day.
- **Output**:
  - Saves predictions to `bitcoin_predictions_90d_with_sentiment.csv` and `bitcoin_predictions_90d_no_sentiment.csv`.
  - Plots combined historical (actual + predicted) and future predictions in `combined_historical_and_future_prediction.png`.
  - Prints a table of future predictions.

---

### **Key Components**
1. **Libraries**:
   - `pandas`, `numpy`: Data manipulation.
   - `yfinance`: Price data retrieval.
   - `sklearn`: Machine learning tools (XGBoost, scaling, metrics).
   - `ta`: Technical indicators.
   - `arch`: GARCH modeling.
   - `matplotlib`: Visualization.
2. **Models**:
   - **XGBoost**: Gradient boosting for price prediction.
   - **GARCH**: Volatility forecasting.
3. **Sentiment Impact**: Compares models with and without sentiment to assess its predictive value.

---

### **Execution Flow**
1. Fetch and merge data.
2. Engineer features and preprocess data.
3. Train and evaluate XGBoost models.
4. Predict historical prices and forecast future prices.
5. Visualize and save results.

---

### **Potential Improvements**
- **Error Handling**: More robust checks for missing data or API failures.
- **Sentiment Quality**: Validate sentiment data accuracy.
- **Hyperparameters**: Expand the search space or use Bayesian optimization.
- **Real-Time Data**: Integrate live data for continuous updates.

This script is a powerful tool for cryptocurrency price forecasting, blending traditional finance techniques (GARCH) with machine learning and sentiment analysis.
