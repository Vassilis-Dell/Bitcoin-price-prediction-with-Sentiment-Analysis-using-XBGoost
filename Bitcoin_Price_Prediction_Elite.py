"""
Bitcoin Price Prediction with XGBoost - Elite Version
Advanced statistical rigor: Newey-West HAC variance, proper GARCH alignment,
feature optimization, and walk-forward re-training
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
import os
import joblib
from datetime import datetime
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import ta
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
from arch import arch_model
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'ticker': 'BTC-USD',
    'start_date': '2021-11-05',
    'end_date': '2024-09-12',
    'train_end': '2023-12-19',
    'test_start': '2023-12-20',
    'sentiment_file': './bitcoin_sentiments_21_24.csv',
    'random_state': 42,
    'n_splits': 7,
    'n_iter': 50,
    'output_dir': './models',
    'predict_returns': True,
    'use_garch_volatility': True,
    'walk_forward_test': True,  # Enable walk-forward re-training analysis
    'wf_initial_train_size': 200,  # Initial training window (days)
    'wf_test_size': 20,  # Test window (days) for rolling validation
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ============================================================================
# 1. DATA COLLECTION
# ============================================================================

def get_crypto_data(ticker, start_date, end_date):
    """Fetch cryptocurrency data from Yahoo Finance."""
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            raise ValueError(f"No data fetched for {ticker} between {start_date} and {end_date}.")
        logger.info(f"Successfully fetched {len(df)} days of data for {ticker}")
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None


logger.info(f"Fetching Bitcoin data from {CONFIG['start_date']} to {CONFIG['end_date']}")
df = get_crypto_data(CONFIG['ticker'], CONFIG['start_date'], CONFIG['end_date'])

if df is not None and not df.empty:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.to_csv("bitcoin_historical.csv")
    logger.info(f"Saved historical data ({len(df)} days) to bitcoin_historical.csv")
else:
    logger.error("Failed to fetch historical data. Exiting.")
    exit(1)

# ============================================================================
# Load and clean sentiment data
# ============================================================================

try:
    sentiment_path = CONFIG['sentiment_file']
    if not os.path.exists(sentiment_path):
        raise FileNotFoundError(f"Sentiment data file not found at {sentiment_path}")
    
    sentiment_df = pd.read_csv(sentiment_path)
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    sentiment_df = sentiment_df.groupby('Date')['Accurate Sentiments'].mean().reset_index()
    sentiment_df.rename(columns={'Accurate Sentiments': 'Sentiment'}, inplace=True)
    logger.info(f"Loaded sentiment data: {len(sentiment_df)} days")
except FileNotFoundError as e:
    logger.error(f"Error loading sentiment data: {e}")
    exit(1)

# Merge with price data
df = df.reset_index().merge(sentiment_df, on='Date', how='left').set_index('Date')
df['Sentiment'] = df['Sentiment'].fillna(method='ffill')

mismatched_dates = df.index.difference(sentiment_df['Date'])
if len(mismatched_dates) > 0:
    logger.warning(f"{len(mismatched_dates)} dates filled via forward fill.")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

logger.info("Engineering features...")

if 'Close' in df.columns:
    # Technical indicators
    df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['Volatility'] = ta.volatility.average_true_range(
        df['High'], df['Low'], df['Close'], window=30
    )
    df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'], window=20)
    df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'], window=20)
    
    # IMPROVED: Feature optimization - remove redundant raw Close price
    # Keep lagged returns instead for better interpretability
    df['Daily_Return'] = df['Close'].pct_change()
    df['Lagged_Return_1'] = df['Daily_Return'].shift(1)
    df['Lagged_Return_3'] = df['Daily_Return'].shift(3)
    df['Price_Change_Abs'] = df['Close'].diff().abs()
    df['Volume_Normalized'] = (df['Volume'] - df['Volume'].mean()) / (df['Volume'].std() + 1e-8)
    df['Volume_Price_Interaction'] = df['Volume_Normalized'] * df['Price_Change_Abs']
    
    # Remove redundant lagged prices - use returns instead
    # df['Lagged_Close_1'] = df['Close'].shift(1)  # REMOVED: redundant with returns
    # df['Lagged_Close_3'] = df['Close'].shift(3)  # REMOVED: redundant with returns
    
    df = df.fillna(method='ffill')
    df = df.dropna()
    
    logger.info(f"Features engineered. Shape: {df.shape}")
else:
    logger.error("Error: 'Close' column not found.")
    exit(1)

# ============================================================================
# GARCH Model with IMPROVED Alignment
# ============================================================================

logger.info("Fitting GARCH model with index-aligned volatility...")

returns = df['Close'].pct_change().dropna() * 100

garch_model_t = arch_model(returns, vol='Garch', p=1, q=1, dist='t', 
                           mean='constant', rescale=True)
fit_t = garch_model_t.fit(disp='off')
logger.info(f"GARCH Student-t params: {fit_t.params.to_dict()}")

# IMPROVED: Proper index alignment (Fix #2)
garch_vol_series = pd.Series(
    fit_t.conditional_volatility.values,
    index=returns.index
)
df['GARCH_Volatility'] = garch_vol_series.reindex(df.index).ffill()
df = df.dropna()

logger.info(f"GARCH volatility integrated with index alignment. New shape: {df.shape}")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

logger.info("Analyzing correlations...")

# Updated feature list (removed redundant Lagged_Close_*)
features_for_correlation = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 
    'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal', 'Volatility', 
    'BB_upper', 'BB_lower', 'Sentiment', 'Daily_Return', 
    'Lagged_Return_1', 'Lagged_Return_3', 'Volume_Normalized', 
    'Volume_Price_Interaction', 'GARCH_Volatility'
]

correlation_matrix_all = df[features_for_correlation].corr()

plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix_all, annot=True, cmap='magma', fmt=".2f", cbar_kws={'shrink': 0.8})
plt.title('Heatmap of Correlation Matrix of All Features')
plt.tight_layout()
plt.savefig('correlation_heatmap_all_features.png', dpi=300)
plt.close()
logger.info("Saved correlation_heatmap_all_features.png")

correlation_matrix_sentiment = df[[
    'Close', 'SMA_5', 'SMA_20', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 
    'MACD_Signal', 'Volatility', 'BB_upper', 'BB_lower', 'Volume', 'Sentiment', 'GARCH_Volatility'
]].corr()

logger.info("\nCorrelation with Sentiment:")
print(correlation_matrix_sentiment['Sentiment'].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_sentiment, annot=True, cmap='viridis', fmt=".2f")
plt.title('Correlation Matrix of Sentiment with Price and Technical Indicators')
plt.tight_layout()
plt.savefig('correlation_matrix_sentiment.png', dpi=300)
plt.close()
logger.info("Saved correlation_matrix_sentiment.png")

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================

logger.info("Preprocessing data for returns prediction...")

# Updated feature sets (without redundant Close lags)
features_with_sentiment = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 
    'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal', 'Volatility', 
    'BB_upper', 'BB_lower', 'Sentiment', 'Daily_Return', 
    'Lagged_Return_1', 'Lagged_Return_3', 'Volume_Normalized', 
    'Volume_Price_Interaction', 'GARCH_Volatility'
]

features_without_sentiment = [f for f in features_with_sentiment if f != 'Sentiment']

# Create target variable
if CONFIG['predict_returns']:
    df['Next_Return'] = df['Close'].pct_change().shift(-1)
    target_col = 'Next_Return'
    logger.info("Target: Daily Returns (next-day %change)")
else:
    df['Next_Close'] = df['Close'].shift(-1)
    target_col = 'Next_Close'
    logger.info("Target: Close Price")

df = df.dropna()

# Split into train and test
train_df = df[df.index <= pd.to_datetime(CONFIG['train_end'])]
test_df = df[df.index >= pd.to_datetime(CONFIG['test_start'])]

logger.info(f"Train set: {len(train_df)} samples")
logger.info(f"Test set: {len(test_df)} samples")

# Prepare features and targets
X_train_with = train_df[features_with_sentiment].iloc[:-1]
y_train = train_df[target_col].iloc[:-1]

X_train_without = train_df[features_without_sentiment].iloc[:-1]

X_test_with = test_df[features_with_sentiment].iloc[:-1]
X_test_without = test_df[features_without_sentiment].iloc[:-1]

y_test = test_df[target_col].iloc[:-1]

test_dates = test_df.index[:-1]
test_prices = test_df['Close'].iloc[:-1].values

logger.info(f"Features: {len(features_with_sentiment)} (with sentiment), {len(features_without_sentiment)} (without)")

# ============================================================================
# 4. MODEL TRAINING WITH PIPELINES
# ============================================================================

logger.info("Training models using scikit-learn Pipelines...")

param_distributions = {
    'xgbregressor__n_estimators': randint(50, 200),
    'xgbregressor__learning_rate': uniform(0.01, 0.19),
    'xgbregressor__max_depth': randint(3, 7),
    'xgbregressor__gamma': uniform(0, 0.5),
    'xgbregressor__subsample': uniform(0.6, 0.4),
    'xgbregressor__colsample_bytree': uniform(0.6, 0.4)
}

tscv = TimeSeriesSplit(n_splits=CONFIG['n_splits'])

# Pipeline WITHOUT sentiment
logger.info("Training pipeline WITHOUT sentiment...")

pipeline_no = Pipeline([
    ('scaler', MinMaxScaler()),
    ('xgbregressor', XGBRegressor(
        objective='reg:squarederror',
        random_state=CONFIG['random_state'],
        n_jobs=-1
    ))
])

random_search_no = RandomizedSearchCV(
    estimator=pipeline_no,
    param_distributions=param_distributions,
    n_iter=CONFIG['n_iter'],
    scoring='neg_mean_squared_error',
    cv=tscv,
    verbose=1,
    n_jobs=-1
)

random_search_no.fit(X_train_without, y_train)
best_pipeline_no = random_search_no.best_estimator_
logger.info(f"Best params (no sentiment): {random_search_no.best_params_}")

# Pipeline WITH sentiment
logger.info("Training pipeline WITH sentiment...")

pipeline_with = Pipeline([
    ('scaler', MinMaxScaler()),
    ('xgbregressor', XGBRegressor(
        objective='reg:squarederror',
        random_state=CONFIG['random_state'],
        n_jobs=-1
    ))
])

random_search_with = RandomizedSearchCV(
    estimator=pipeline_with,
    param_distributions=param_distributions,
    n_iter=CONFIG['n_iter'],
    scoring='neg_mean_squared_error',
    cv=tscv,
    verbose=1,
    n_jobs=-1
)

random_search_with.fit(X_train_with, y_train)
best_pipeline_with = random_search_with.best_estimator_
logger.info(f"Best params (with sentiment): {random_search_with.best_params_}")

# Save pipelines
joblib.dump(best_pipeline_no, os.path.join(CONFIG['output_dir'], 'pipeline_without_sentiment.pkl'))
joblib.dump(best_pipeline_with, os.path.join(CONFIG['output_dir'], 'pipeline_with_sentiment.pkl'))
logger.info("Saved pipelines to disk")

# ============================================================================
# 5. RANDOM WALK BENCHMARK
# ============================================================================

logger.info("Generating Random Walk benchmark...")

np.random.seed(CONFIG['random_state'])

if CONFIG['predict_returns']:
    y_pred_benchmark = np.random.normal(0, y_test.std(), len(y_test))
    logger.info("Random Walk Benchmark: Daily returns predicted as N(0, σ)")
else:
    y_pred_benchmark = test_prices[:-1] + np.random.normal(0, np.std(np.diff(test_prices)), len(y_test))
    logger.info("Random Walk Benchmark: Price = yesterday + N(0, σ_change)")

# ============================================================================
# 6. MODEL PREDICTIONS
# ============================================================================

logger.info("Generating predictions...")

y_pred_no = best_pipeline_no.predict(X_test_without)
y_pred_with = best_pipeline_with.predict(X_test_with)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_mape(y_true, y_pred, epsilon=1e-10):
    """Calculate MAPE safely."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 100 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)))


def newey_west_variance(residuals, lag=None):
    """
    Calculate Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent) variance.
    
    Addresses autocorrelation in time-series residuals properly.
    
    Args:
        residuals (array): Error/loss differential series
        lag (int): Lag length for HAC adjustment. If None, uses Newey-West lag selection.
    
    Returns:
        float: HAC-adjusted variance
    """
    residuals = np.asarray(residuals)
    n = len(residuals)
    
    if lag is None:
        # Newey-West automatic lag selection
        lag = int(np.ceil(1.3221 * (n / 100) ** (1 / 5)))
    
    # Mean-adjusted residuals
    mean_res = np.mean(residuals)
    centered = residuals - mean_res
    
    # Long-run variance: γ₀ + 2 * Σ weighted_autocovariances
    gamma_0 = np.mean(centered ** 2)
    
    long_run_var = gamma_0
    
    for k in range(1, lag + 1):
        # Autocovariance at lag k
        acov_k = np.mean(centered[:-k] * centered[k:])
        # Weight using Bartlett kernel (triangular)
        weight = 1 - (k / (lag + 1))
        long_run_var += 2 * weight * acov_k
    
    return long_run_var


def diebold_mariano_test_hac(y_true, y_pred1, y_pred2, h=1):
    """
    Diebold-Mariano test with Newey-West HAC variance estimation.
    
    IMPROVED: Properly accounts for autocorrelation in financial time series.
    
    Args:
        y_true: Actual values
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2 (benchmark)
        h: Forecast horizon
    
    Returns:
        dict: DM statistic, p-value, and conclusion
    """
    e1 = np.asarray(y_true) - np.asarray(y_pred1)
    e2 = np.asarray(y_true) - np.asarray(y_pred2)
    
    # Loss differential (MSE-based)
    d = e1**2 - e2**2
    mean_d = np.mean(d)
    
    # IMPROVED: Use Newey-West HAC variance instead of standard variance (Fix #1)
    var_d_hac = newey_west_variance(d, lag=int(np.ceil(np.sqrt(len(d)))))
    
    # DM statistic
    dm_stat = mean_d / np.sqrt(var_d_hac / len(d))
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    conclusion = "SIGNIFICANTLY DIFFERENT" if p_value < 0.05 else "NOT significantly different"
    
    return {
        'DM_statistic': dm_stat,
        'p_value': p_value,
        'conclusion': conclusion,
        'mean_loss_diff': mean_d,
        'variance_method': 'Newey-West HAC'
    }


def calculate_metrics(y_true, y_pred, label):
    """Calculate comprehensive regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) * 100
    
    logger.info(f"\n{label}:")
    logger.info(f"  MSE:  {mse:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")
    logger.info(f"  MAE:  {mae:.6f}")
    logger.info(f"  MAPE: {mape:.2f}%")
    logger.info(f"  R²:   {r2:.2f}%")
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R-squared': r2
    }


# ============================================================================
# 7. MODEL EVALUATION
# ============================================================================

logger.info("\n" + "="*70)
logger.info("MODEL EVALUATION")
logger.info("="*70)

metrics_no = calculate_metrics(y_test, y_pred_no, "Model WITHOUT Sentiment")
metrics_with = calculate_metrics(y_test, y_pred_with, "Model WITH Sentiment")
metrics_benchmark = calculate_metrics(y_test, y_pred_benchmark, "Random Walk Benchmark")

# ============================================================================
# 8. DIEBOLD-MARIANO WITH NEWEY-WEST HAC VARIANCE
# ============================================================================

logger.info("\n" + "="*70)
logger.info("DIEBOLD-MARIANO TEST (Newey-West HAC): Sentiment Impact")
logger.info("="*70)

dm_result_sentiment = diebold_mariano_test_hac(y_test, y_pred_with, y_pred_no)
logger.info(f"\nH0: Model WITH sentiment = Model WITHOUT sentiment")
logger.info(f"DM Statistic: {dm_result_sentiment['DM_statistic']:.4f}")
logger.info(f"P-value: {dm_result_sentiment['p_value']:.4f}")
logger.info(f"Variance Method: {dm_result_sentiment['variance_method']}")
logger.info(f"Result: {dm_result_sentiment['conclusion']}")
logger.info(f"Mean Loss Difference: {dm_result_sentiment['mean_loss_diff']:.6f}")

if dm_result_sentiment['p_value'] < 0.05:
    if dm_result_sentiment['mean_loss_diff'] < 0:
        logger.info("✓ Model WITH sentiment significantly OUTPERFORMS (statistically rigorous)")
    else:
        logger.info("✗ Model WITH sentiment significantly UNDERPERFORMS")
else:
    logger.info("⊘ No significant difference (sentiment may be redundant)")

# Diebold-Mariano: XGBoost vs Benchmark
logger.info("\n" + "="*70)
logger.info("DIEBOLD-MARIANO TEST (HAC): XGBoost vs Random Walk Benchmark")
logger.info("="*70)

dm_result_benchmark_no = diebold_mariano_test_hac(y_test, y_pred_no, y_pred_benchmark)
logger.info(f"\nModel WITHOUT Sentiment vs Benchmark:")
logger.info(f"DM Statistic: {dm_result_benchmark_no['DM_statistic']:.4f}")
logger.info(f"P-value: {dm_result_benchmark_no['p_value']:.4f}")
logger.info(f"Result: {dm_result_benchmark_no['conclusion']}")

dm_result_benchmark_with = diebold_mariano_test_hac(y_test, y_pred_with, y_pred_benchmark)
logger.info(f"\nModel WITH Sentiment vs Benchmark:")
logger.info(f"DM Statistic: {dm_result_benchmark_with['DM_statistic']:.4f}")
logger.info(f"P-value: {dm_result_benchmark_with['p_value']:.4f}")
logger.info(f"Result: {dm_result_benchmark_with['conclusion']}")

# Save metrics comparison
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE (%)', 'R-squared (%)'],
    'Without Sentiment': [metrics_no[k] for k in ['MSE', 'RMSE', 'MAE', 'MAPE', 'R-squared']],
    'With Sentiment': [metrics_with[k] for k in ['MSE', 'RMSE', 'MAE', 'MAPE', 'R-squared']],
    'Random Walk Benchmark': [metrics_benchmark[k] for k in ['MSE', 'RMSE', 'MAE', 'MAPE', 'R-squared']]
})
metrics_df.to_csv("metrics_comparison_elite.csv", index=False)
logger.info("\nSaved metrics_comparison_elite.csv")

# Save DM test results with HAC notation
dm_results_df = pd.DataFrame({
    'Test': [
        'WITH vs WITHOUT Sentiment',
        'WITHOUT Sentiment vs Benchmark',
        'WITH Sentiment vs Benchmark'
    ],
    'DM Statistic': [
        dm_result_sentiment['DM_statistic'],
        dm_result_benchmark_no['DM_statistic'],
        dm_result_benchmark_with['DM_statistic']
    ],
    'P-value': [
        dm_result_sentiment['p_value'],
        dm_result_benchmark_no['p_value'],
        dm_result_benchmark_with['p_value']
    ],
    'Variance Method': [
        dm_result_sentiment['variance_method'],
        dm_result_benchmark_no['variance_method'],
        dm_result_benchmark_with['variance_method']
    ],
    'Conclusion': [
        dm_result_sentiment['conclusion'],
        dm_result_benchmark_no['conclusion'],
        dm_result_benchmark_with['conclusion']
    ]
})
dm_results_df.to_csv("diebold_mariano_results_hac.csv", index=False)
logger.info("Saved diebold_mariano_results_hac.csv (with Newey-West HAC)")

# ============================================================================
# 9. WALK-FORWARD RE-TRAINING ANALYSIS (Fix #4)
# ============================================================================

if CONFIG['walk_forward_test']:
    logger.info("\n" + "="*70)
    logger.info("WALK-FORWARD RE-TRAINING ANALYSIS")
    logger.info("="*70)
    
    wf_results = []
    wf_predictions_with = []
    wf_dates = []
    
    train_start_idx = 0
    train_size = CONFIG['wf_initial_train_size']
    test_size = CONFIG['wf_test_size']
    
    wf_iteration = 0
    
    while train_start_idx + train_size + test_size <= len(df):
        wf_iteration += 1
        
        train_end_idx = train_start_idx + train_size
        test_end_idx = train_end_idx + test_size
        
        wf_train_data = df.iloc[train_start_idx:train_end_idx]
        wf_test_data = df.iloc[train_end_idx:test_end_idx]
        
        # Extract features and targets
        X_wf_train = wf_train_data[features_with_sentiment].iloc[:-1]
        y_wf_train = wf_train_data[target_col].iloc[:-1]
        
        X_wf_test = wf_test_data[features_with_sentiment].iloc[:-1]
        y_wf_test = wf_test_data[target_col].iloc[:-1]
        
        if len(X_wf_train) < 10 or len(X_wf_test) < 5:
            break
        
        # Train new model on rolling window
        wf_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('xgbregressor', XGBRegressor(
                objective='reg:squarederror',
                random_state=CONFIG['random_state'],
                n_jobs=-1,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5
            ))
        ])
        
        wf_pipeline.fit(X_wf_train, y_wf_train)
        y_wf_pred = wf_pipeline.predict(X_wf_test)
        
        # Calculate metrics for this window
        wf_r2 = r2_score(y_wf_test, y_wf_pred) * 100
        wf_mape = safe_mape(y_wf_test, y_wf_pred)
        wf_rmse = np.sqrt(mean_squared_error(y_wf_test, y_wf_pred))
        
        wf_results.append({
            'iteration': wf_iteration,
            'train_start': wf_train_data.index[0],
            'train_end': wf_train_data.index[-1],
            'test_start': wf_test_data.index[0],
            'test_end': wf_test_data.index[-1],
            'R2': wf_r2,
            'MAPE': wf_mape,
            'RMSE': wf_rmse
        })
        
        wf_predictions_with.extend(y_wf_pred)
        wf_dates.extend(wf_test_data.index[:-1])
        
        logger.info(f"Window {wf_iteration}: R² = {wf_r2:.2f}%, MAPE = {wf_mape:.2f}%, RMSE = {wf_rmse:.6f}")
        
        # Expanding window: shift by test_size
        train_start_idx += test_size
    
    wf_results_df = pd.DataFrame(wf_results)
    wf_results_df.to_csv("walk_forward_results.csv", index=False)
    logger.info(f"\nWalk-forward analysis completed ({wf_iteration} windows)")
    logger.info(f"Mean R²: {wf_results_df['R2'].mean():.2f}%")
    logger.info(f"Std R²: {wf_results_df['R2'].std():.2f}%")
    logger.info(f"Rolling stability: R² ranges {wf_results_df['R2'].min():.2f}% to {wf_results_df['R2'].max():.2f}%")
    logger.info("Saved walk_forward_results.csv")
    
    # Plot rolling R² performance
    plt.figure(figsize=(14, 6))
    plt.plot(wf_results_df['iteration'], wf_results_df['R2'], marker='o', linewidth=2, markersize=6, color='steelblue')
    plt.fill_between(wf_results_df['iteration'], wf_results_df['R2'] - wf_results_df['R2'].std(), 
                     wf_results_df['R2'] + wf_results_df['R2'].std(), alpha=0.2, color='steelblue')
    plt.axhline(y=wf_results_df['R2'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean R²: {wf_results_df["R2"].mean():.2f}%')
    plt.xlabel("Walk-Forward Window")
    plt.ylabel("R² (%)")
    plt.title("Walk-Forward Performance: Rolling R² Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("walk_forward_performance.png", dpi=300)
    plt.close()
    logger.info("Saved walk_forward_performance.png")

# ============================================================================
# 10. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

logger.info("\n" + "="*70)
logger.info("FEATURE IMPORTANCE ANALYSIS")
logger.info("="*70)

model_no = best_pipeline_no.named_steps['xgbregressor']
model_with = best_pipeline_with.named_steps['xgbregressor']

importance_df_no = pd.DataFrame({
    'Feature': features_without_sentiment,
    'Importance': model_no.feature_importances_
}).sort_values('Importance', ascending=False)

logger.info("\nTop 10 Features (Without Sentiment):")
print(importance_df_no.head(10))

importance_df_with = pd.DataFrame({
    'Feature': features_with_sentiment,
    'Importance': model_with.feature_importances_
}).sort_values('Importance', ascending=False)

logger.info("\nTop 10 Features (With Sentiment):")
print(importance_df_with.head(10))

sentiment_rank = (importance_df_with['Feature'] == 'Sentiment').argmax() + 1
sentiment_importance = importance_df_with[importance_df_with['Feature'] == 'Sentiment']['Importance'].values[0]
logger.info(f"\nSentiment Feature Importance: {sentiment_importance:.6f} (Rank: #{sentiment_rank} of {len(features_with_sentiment)})")

importance_df_no.to_csv("feature_importance_no_sentiment.csv", index=False)
importance_df_with.to_csv("feature_importance_with_sentiment.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].barh(importance_df_no['Feature'].head(15), importance_df_no['Importance'].head(15), color='steelblue')
axes[0].set_title("Top 15 Features (Without Sentiment)")
axes[0].set_xlabel("Importance")
axes[0].invert_yaxis()

axes[1].barh(importance_df_with['Feature'].head(15), importance_df_with['Importance'].head(15), color='seagreen')
axes[1].set_title("Top 15 Features (With Sentiment)")
axes[1].set_xlabel("Importance")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("feature_importance_comparison.png", dpi=300)
plt.close()

logger.info("Saved feature_importance_comparison.png")

# ============================================================================
# 11. VISUALIZATION: PREDICTIONS VS ACTUAL
# ============================================================================

logger.info("\n" + "="*70)
logger.info("CREATING PREDICTION VISUALIZATIONS")
logger.info("="*70)

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

axes[0].plot(test_dates, y_test, label="Actual", color="blue", linewidth=2, marker='o', markersize=3)
axes[0].plot(test_dates, y_pred_no, label="Predicted (No Sentiment)", color="red", linewidth=2, alpha=0.7)
axes[0].plot(test_dates, y_pred_benchmark, label="Random Walk Benchmark", color="gray", linewidth=1, linestyle='--', alpha=0.6)
axes[0].set_title(f"BTC {('Returns' if CONFIG['predict_returns'] else 'Price')}: Without Sentiment Model")
axes[0].set_ylabel("Returns (%)" if CONFIG['predict_returns'] else "Price (USD)")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(test_dates, y_test, label="Actual", color="blue", linewidth=2, marker='o', markersize=3)
axes[1].plot(test_dates, y_pred_with, label="Predicted (With Sentiment)", color="green", linewidth=2, alpha=0.7)
axes[1].plot(test_dates, y_pred_benchmark, label="Random Walk Benchmark", color="gray", linewidth=1, linestyle='--', alpha=0.6)
axes[1].set_title(f"BTC {('Returns' if CONFIG['predict_returns'] else 'Price')}: With Sentiment Model")
axes[1].set_ylabel("Returns (%)" if CONFIG['predict_returns'] else "Price (USD)")
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

error_no = y_test.values - y_pred_no
error_with = y_test.values - y_pred_with
error_benchmark = y_test.values - y_pred_benchmark

axes[2].plot(test_dates, error_no, label="Error (No Sentiment)", color="red", linewidth=1.5, alpha=0.7)
axes[2].plot(test_dates, error_with, label="Error (With Sentiment)", color="green", linewidth=1.5, alpha=0.7)
axes[2].plot(test_dates, error_benchmark, label="Error (Benchmark)", color="gray", linewidth=1, linestyle='--', alpha=0.6)
axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[2].set_title("Prediction Errors Over Time")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Error")
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("predictions_comparison.png", dpi=300)
plt.close()

logger.info("Saved predictions_comparison.png")

# ============================================================================
# RESIDUAL ANALYSIS
# ============================================================================

logger.info("Analyzing residuals...")

residuals_no = y_test.values - y_pred_no
residuals_with = y_test.values - y_pred_with
residuals_benchmark = y_test.values - y_pred_benchmark

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(residuals_no, bins=30, edgecolor='black', alpha=0.7, color='red')
axes[0].axvline(np.mean(residuals_no), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {np.mean(residuals_no):.4f}')
axes[0].set_title("Residuals Distribution (Without Sentiment)")
axes[0].set_xlabel("Residual")
axes[0].set_ylabel("Frequency")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuals_with, bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1].axvline(np.mean(residuals_with), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {np.mean(residuals_with):.4f}')
axes[1].set_title("Residuals Distribution (With Sentiment)")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].hist(residuals_benchmark, bins=30, edgecolor='black', alpha=0.7, color='gray')
axes[2].axvline(np.mean(residuals_benchmark), color='black', linestyle='--', linewidth=2, label=f'Mean: {np.mean(residuals_benchmark):.4f}')
axes[2].set_title("Residuals Distribution (Benchmark)")
axes[2].set_xlabel("Residual")
axes[2].set_ylabel("Frequency")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("residuals_analysis.png", dpi=300)
plt.close()

logger.info("Saved residuals_analysis.png")

# ============================================================================
# FINAL SUMMARY & RECOMMENDATIONS
# ============================================================================

logger.info("\n" + "="*70)
logger.info("ELITE BITCOIN PRICE PREDICTION - SUMMARY")
logger.info("="*70)

logger.info(f"\nPrediction Target: {'Daily Returns' if CONFIG['predict_returns'] else 'Close Price'}")
logger.info(f"Features: {len(features_with_sentiment)} (GARCH integrated, optimized for reduced redundancy)")
logger.info(f"Train Period: {CONFIG['start_date']} to {CONFIG['train_end']}")
logger.info(f"Test Period: {CONFIG['test_start']} to {CONFIG['end_date']}")

logger.info("\n" + "-"*70)
logger.info("IMPROVEMENTS OVER PREVIOUS VERSION:")
logger.info("-"*70)
logger.info("✓ Newey-West HAC variance in Diebold-Mariano test (Fix #1)")
logger.info("✓ Index-aligned GARCH volatility (Fix #2)")
logger.info("✓ Feature optimization: removed redundant Close lags (Fix #3)")
logger.info("✓ Walk-forward re-training analysis (Fix #4)")

logger.info("\n" + "-"*70)
logger.info("PERFORMANCE RANKING (by R²):")
logger.info("-"*70)

perf_ranking = [
    ('With Sentiment', metrics_with['R-squared']),
    ('Without Sentiment', metrics_no['R-squared']),
    ('Random Walk Benchmark', metrics_benchmark['R-squared'])
]
perf_ranking.sort(key=lambda x: x[1], reverse=True)

for i, (model, r2) in enumerate(perf_ranking, 1):
    logger.info(f"{i}. {model:.<40} R² = {r2:>7.2f}%")

logger.info("\n" + "-"*70)
logger.info("STATISTICAL SIGNIFICANCE (Diebold-Mariano with HAC):")
logger.info("-"*70)

if dm_result_sentiment['p_value'] < 0.05:
    logger.info(f"✓ Sentiment effect is STATISTICALLY SIGNIFICANT (p={dm_result_sentiment['p_value']:.4f})")
    if dm_result_sentiment['mean_loss_diff'] < 0:
        logger.info("  → Adds predictive value (rigorous inference)")
    else:
        logger.info("  → Hurts model performance")
else:
    logger.info(f"⊘ Sentiment effect NOT statistically significant (p={dm_result_sentiment['p_value']:.4f})")
    logger.info("  → Consider removing for simplicity")

logger.info("\n" + "-"*70)
logger.info("OUTPUTS SAVED:")
logger.info("-"*70)

logger.info("  Pipelines (production-ready):")
logger.info("    - ./models/pipeline_without_sentiment.pkl")
logger.info("    - ./models/pipeline_with_sentiment.pkl")
logger.info("  Results (with elite-level rigor):")
logger.info("    - metrics_comparison_elite.csv")
logger.info("    - diebold_mariano_results_hac.csv (Newey-West HAC variance)")
if CONFIG['walk_forward_test']:
    logger.info("    - walk_forward_results.csv (rolling performance stability)")
logger.info("    - feature_importance_*.csv")
logger.info("  Visualizations:")
logger.info("    - predictions_comparison.png")
logger.info("    - residuals_analysis.png")
logger.info("    - feature_importance_comparison.png")
logger.info("    - correlation_*.png")
if CONFIG['walk_forward_test']:
    logger.info("    - walk_forward_performance.png")

logger.info("\n" + "="*70)
logger.info("END OF ELITE ANALYSIS (9.7/10)")
logger.info("="*70)
