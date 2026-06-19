import logging

import glob
def get_latest_dataset(symbol, data_dir='data'):
    files = glob.glob(os.path.join(data_dir, f"{symbol}_*_to_*.csv"))
    if not files:
        old_file = os.path.join(data_dir, f"{symbol}_processed_data.csv")
        if os.path.exists(old_file): return old_file
        raise FileNotFoundError(f"No dataset found for {symbol}")
    return sorted(files)[-1]
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import os
import gc
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Ignore warnings to keep logs clean
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logs

# =============================================================================
# DATA PREPARATION (From lstm.ipynb)
# =============================================================================
def prepare_data(symbol, data_dir='data'):
    file_path = get_latest_dataset(symbol, data_dir)
    logger.info(f"\n{'='*50}\n--- Preparing Data for {symbol} ---\n{'='*50}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Skipping {symbol}.")

    df = pd.read_csv(file_path)
    df['Stock_Timestamp'] = pd.to_datetime(df['Stock_Timestamp'], utc=True)
    df = df.sort_values('Stock_Timestamp').reset_index(drop=True)

    # Feature Engineering (Exact mapping from lstm.ipynb)
    sentiment_map = {'positive': 1, 'neutral': 0, 'mixed': 0, 'negative': -1}
    if 'Sentiment' in df.columns:
        df['Sentiment_Score'] = df['Sentiment'].map(sentiment_map).fillna(0)
        df['Decayed_Sentiment'] = df['Sentiment_Score'] / np.log1p(df.get('News_Age_Minutes', 0) + 1)
    else:
        df['Decayed_Sentiment'] = 0

    df['Close_to_SMA50'] = df['Close'] / df['SMA_50']
    df['Close_to_EMA10'] = df['Close'] / df['EMA_10']
    df['Close_to_Kalman'] = df['Close'] / df['kalman_close']

    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Relative_Volume'] = df['Volume'] / df['Vol_SMA20']

    range_denom = df['rolling_high_20'] - df['rolling_low_20']
    df['Price_Range_Position'] = np.where(
        range_denom == 0, 0.5, (df['Close'] - df['rolling_low_20']) / range_denom
    )

    minutes_since_midnight = df['Stock_Timestamp'].dt.hour * 60 + df['Stock_Timestamp'].dt.minute
    day_minutes = 1440
    df['Time_Sin'] = np.sin(2 * np.pi * minutes_since_midnight / day_minutes)
    df['Time_Cos'] = np.cos(2 * np.pi * minutes_since_midnight / day_minutes)

    df['Trans_SMA20'] = df['Transactions'].rolling(window=20).mean()
    df['Relative_Transactions'] = df['Transactions'] / df['Trans_SMA20']

    df['Returns_1m'] = df['Close'].pct_change()
    df['Realized_Volatility'] = df['Returns_1m'].rolling(window=20).std()

    # Target: Predict if the next minute's close is higher than the current close
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    feature_cols = [
        'RSI_14', 'MACD_12_26_9', 'kalman_diff',    
        'Close_to_SMA50', 'Close_to_EMA10', 'Close_to_Kalman', 
        'Relative_Volume', 'Price_Range_Position',     
        'Relative_Transactions', 'Realized_Volatility',
        'Decayed_Sentiment',                           
        'Time_Sin', 'Time_Cos'                         
    ]

    # Drop NaNs and keep raw close and timestamp for simulation later
    model_df = df[['Stock_Timestamp', 'Open', 'Close'] + feature_cols + ['Target']].dropna().reset_index(drop=True)
    
    logger.info(f"[{symbol}] Data Prepared. Final Shape: {model_df.shape}")
    return model_df, feature_cols

# =============================================================================
# SEQUENCE GENERATION
# =============================================================================
def create_sequences(data, window_size=20):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, :-1]) # All columns except Target
        y.append(data[i, -1])                # Last column is Target
    return np.array(X), np.array(y)

# =============================================================================
# INDIVIDUAL PIPELINE RUNNER
# =============================================================================
def run_pipeline_for_symbol(symbol):
    def get_next_version(symbol, model_name="lstm", mode="Normal"):
        date_str = datetime.now().strftime('%Y%m%d')
        pattern = os.path.join('saved_models', f'{symbol}_{model_name}_{mode}_{date_str}_v*.pth')
        files = glob.glob(pattern)
        pattern_ckpt = os.path.join('saved_models', f'{symbol}_{model_name}_{mode}_{date_str}_v*.ckpt')
        files.extend(glob.glob(pattern_ckpt))
        if not files: return 1
        versions = []
        for f in files:
            try:
                v = int(re.search(r'_v(\d+)\\.', f).group(1))
                versions.append(v)
            except: pass
        return max(versions) + 1 if versions else 1

    date_str = datetime.now().strftime('%Y%m%d')
    version = get_next_version(symbol)
    base_output_name = f"{symbol}_lstm_Normal_{date_str}_v{version}"

    try:
        model_df, feature_cols = prepare_data(symbol, data_dir='data')
    except FileNotFoundError as e:
        logger.info(f"❌ {e}")
        return  # Skip to the next symbol

    # 2. Scaling & Shaping (From lstm.ipynb)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(model_df[feature_cols])
    final_data = np.column_stack((scaled_features, model_df['Target'].values))

    # Splits (70 / 20 / 10)
    n = len(final_data)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.90) 

    train_data = final_data[:train_idx]
    val_data = final_data[train_idx:val_idx]
    test_data = final_data[val_idx:]

    window_size = 20
    X_train, y_train = create_sequences(train_data, window_size)
    X_val, y_val = create_sequences(val_data, window_size)
    X_test, y_test = create_sequences(test_data, window_size)

    logger.info(f"Splits for {symbol}: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 3. Initialize Model (TensorFlow LSTM identical to your notebook)
    logger.info(f"\n--- Initializing LSTM Model for {symbol} ---")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 4. Configure Callbacks
    checkpoint_path = os.path.join("saved_models", f"{symbol}_lstm_best.keras")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0)
    ]

    # 5. Train Model
    logger.info(f"\n--- Starting Training for {symbol} ---")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=64, 
        callbacks=callbacks,
        verbose=1 # Prints standard TF epochs cleanly
    )
    logger.info(f"\n✅ Training Complete. Best {symbol} model saved at: {checkpoint_path}")

    # =========================================================================
    # EVALUATION & VISUALIZATION
    # =========================================================================
    logger.info(f"\n--- Generating Statistical Performance for {symbol} (10% Test Set) ---")
    
    # Load the best model specifically
    best_model = tf.keras.models.load_model(checkpoint_path)
    
    # Predictions
    y_pred_probs = best_model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Classification Report
    cr = classification_report(y_test, y_pred, labels=[0, 1], target_names=['Down', 'Up'], zero_division=0)
    logger.info(cr)
    
    # Save Report to file dynamically
    report_path = os.path.join('visualizations', f'{base_output_name}_report.txt')
    with open(report_path, "w") as f:
        f.write(f"--- Final Statistical Performance for {symbol} (10% Test Set) ---\n")
        f.write(cr)

    # Confusion Matrix dynamically
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Down', 'Up'], cmap='Blues')
    plt.title(f"{symbol} LSTM Confusion Matrix")
    cm_path = os.path.join('visualizations', f'{base_output_name}_cm.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Confusion Matrix saved to: {cm_path}")

    # =========================================================================
    # COMPOUNDING SIMULATION
    # =========================================================================
    logger.info(f"\n--- Running Compounding Simulation for {symbol} ---")
    
    # Map back original DataFrame timestamps/prices for the test set
    # Test indices start at val_idx + window_size
    test_start_idx = val_idx + window_size
    sim_df = model_df.iloc[test_start_idx:].copy().reset_index(drop=True)
    sim_df['Signal'] = y_pred

    initial_capital = 10000.0
    current_balance = initial_capital
    cost_per_share = 0.009

    # Using standard logic mapping
    sim_df['Entry_Price'] = sim_df['Open']
    sim_df['Exit_Price'] = sim_df['Open'].shift(-1)

    balances = []
    for i in range(len(sim_df)):
        row = sim_df.iloc[i]

        if row['Signal'] == 1 and not np.isnan(row['Exit_Price']):
            num_shares = current_balance // row['Entry_Price']
            if num_shares > 0:
                trade_profit = (num_shares * (row['Exit_Price'] - row['Entry_Price'])) - (num_shares * cost_per_share)
                current_balance += trade_profit
        balances.append(current_balance)

    sim_df['Portfolio_Value'] = balances

    logger.info(f"[{symbol}] Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"[{symbol}] Final Balance:   ${current_balance:,.2f}")
    logger.info(f"[{symbol}] Total Return:    {((current_balance - initial_capital)/initial_capital)*100:,.2f}%")

    # Plot Equity Curve dynamically
    plt.figure(figsize=(12, 5))
    plt.plot(sim_df['Stock_Timestamp'], sim_df['Portfolio_Value'], color='green', linewidth=2)
    plt.axhline(initial_capital, color='red', linestyle='--', label='Initial Capital')
    plt.title(f"{symbol} LSTM Equity Curve: Full Reinvestment (Compounding)")
    plt.ylabel("Account Balance ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    equity_path = os.path.join('visualizations', f'{base_output_name}_equity.png')
    plt.savefig(equity_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Equity Curve saved to: {equity_path}\n")

    
    # Save Actual vs Predicted CSV
    csv_path = os.path.join('visualizations', f'{base_output_name}_actual_vs_pred.csv')
    try:
        if 'Stock_Timestamp' in sim_df.columns:
            ts = sim_df['Stock_Timestamp']
        else:
            ts = sim_df.index
        df_csv = pd.DataFrame({
            'Timestamp': ts,
            'Actual': y_true if 'y_true' in locals() else sim_df['Exit_Price'] - sim_df['Entry_Price'], # rough fallback
            'Predicted': y_pred
        })
        df_csv.to_csv(csv_path, index=False)
        logger.info(f"✅ Predictions CSV saved to: {csv_path}")
    except Exception as e:
        logger.error(f"Could not save Predictions CSV: {e}")

    # Clean up memory before the next loop iteration
    del model, best_model, X_train, y_train, X_val, y_val, X_test, y_test
    tf.keras.backend.clear_session()
    gc.collect()


# =============================================================================
# MAIN LOOP
# =============================================================================

def run_training(symbol, mode="Normal"):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    logger.info(f"Starting training for {symbol} in {mode} mode...")
    run_pipeline_for_symbol(symbol)
    logger.info(f"✅ {symbol} processed successfully!")

if __name__ == "__main__":
    run_training("MSFT")
