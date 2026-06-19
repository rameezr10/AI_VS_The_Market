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
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Ignore warnings to keep logs clean
warnings.filterwarnings("ignore")

# =============================================================================
# DATA PREPARATION (From GRU.ipynb logic)
# =============================================================================
def prepare_data(symbol, data_dir='data', window_size=20):
    file_path = get_latest_dataset(symbol, data_dir)
    logger.info(f"\n{'='*50}\n--- Preparing Data for {symbol} ---\n{'='*50}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Skipping {symbol}.")

    df = pd.read_csv(file_path)
    df['Stock_Timestamp'] = pd.to_datetime(df['Stock_Timestamp'], utc=True)
    df = df.sort_values('Stock_Timestamp').reset_index(drop=True)

    # --- Feature Engineering (Specific to GRU.ipynb) ---
    sentiment_map = {'positive': 1, 'neutral': 0, 'mixed': 0, 'negative': -1}
    df['Sentiment_Score'] = df['Sentiment'].map(sentiment_map).fillna(0)
    df['Decayed_Sentiment'] = df['Sentiment_Score'] / np.log1p(df['News_Age_Minutes'] + 1)

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
    df['Time_Sin'] = np.sin(2 * np.pi * minutes_since_midnight / 1440)
    df['Time_Cos'] = np.cos(2 * np.pi * minutes_since_midnight / 1440)

    df['Trans_SMA20'] = df['Transactions'].rolling(window=20).mean()
    df['Relative_Transactions'] = df['Transactions'] / df['Trans_SMA20']

    df['Returns_1m'] = df['Close'].pct_change()
    df['Realized_Volatility'] = df['Returns_1m'].rolling(window=20).std()

    # Target: 1 if the NEXT Close is higher than current Close
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    feature_cols = [
        'RSI_14', 'MACD_12_26_9', 'kalman_diff',    
        'Close_to_SMA50', 'Close_to_EMA10', 'Close_to_Kalman', 
        'Relative_Volume', 'Price_Range_Position',     
        'Relative_Transactions', 'Realized_Volatility',
        'Decayed_Sentiment',                           
        'Time_Sin', 'Time_Cos'                         
    ]

    model_df = df[feature_cols + ['Target']].dropna()
    raw_test_df = df.iloc[model_df.index].copy() # Keep for financial simulation
    
    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(model_df[feature_cols])
    final_data = np.column_stack((scaled_features, model_df['Target'].values))

    # Sequencing
    def create_sequences(data, window):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i, :-1])
            y.append(data[i, -1])
        return np.array(X), np.array(y)

    n = len(final_data)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.90)

    X_all, y_all = create_sequences(final_data, window_size)
    
    # Adjust indices for the window offset
    X_train, y_train = X_all[:train_idx-window_size], y_all[:train_idx-window_size]
    X_val, y_val = X_all[train_idx-window_size:val_idx-window_size], y_all[train_idx-window_size:val_idx-window_size]
    X_test, y_test = X_all[val_idx-window_size:], y_all[val_idx-window_size:]
    
    # Financial test set alignment
    test_sim_df = raw_test_df.iloc[val_idx:].copy()

    return X_train, y_train, X_val, y_val, X_test, y_test, test_sim_df, len(feature_cols)

# =============================================================================
# INDIVIDUAL PIPELINE RUNNER
# =============================================================================
def run_pipeline_for_symbol(symbol):
    def get_next_version(symbol, model_name="gru", mode="Normal"):
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
    base_output_name = f"{symbol}_gru_Normal_{date_str}_v{version}"

    window_size = 20
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, sim_df, input_dim = prepare_data(symbol)
    except FileNotFoundError as e:
        logger.info(f"❌ {e}")
        return

    # 1. Build Model (From GRU.ipynb)
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(window_size, input_dim)),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 2. Training
    logger.info(f"\n--- Training GRU Model for {symbol} ---")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0 # Clean logs
    )

    # 3. Save Model
    model_path = os.path.join('saved_models', f'{symbol}_GRU_model.h5')
    model.save(model_path)
    logger.info(f"✅ Model saved at: {model_path}")

    # 4. Evaluation
    logger.info(f"\n--- Generating Performance Results for {symbol} ---")
    probs = model.predict(X_test, verbose=0)
    y_pred = (probs > 0.5).astype(int).flatten()

    # Classification Report
    cr = classification_report(y_test, y_pred, target_names=['Down', 'Up'], zero_division=0)
    logger.info(cr)
    report_path = os.path.join('visualizations', f'{base_output_name}_report.txt')
    with open(report_path, "w") as f:
        f.write(cr)

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Down', 'Up'], cmap='Blues')
    plt.title(f"{symbol} GRU Confusion Matrix")
    cm_path = os.path.join('visualizations', f'{base_output_name}_cm.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Financial Simulation (From GRU.ipynb)
    logger.info(f"--- Running Financial Simulation for {symbol} ---")
    initial_capital = 10000.0
    current_balance = initial_capital
    cost_per_share = 0.009
    
    # Align prediction with simulation dataframe
    sim_df = sim_df.iloc[:len(y_pred)].copy() 
    sim_df['Signal'] = y_pred
    
    balances = []
    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        # Logic: If prediction is Up (1), buy at current Close, sell at next Close (Target alignment)
        if row['Signal'] == 1:
            num_shares = current_balance // row['Close']
            if num_shares > 0:
                # Profit calculation based on the Target logic (Next Close > Current Close)
                # Since Target was (Close.shift(-1) > Close), we simulate that gain here
                price_change = row['Close'] * (row['Returns_1m'] if i > 0 else 0) # Simplified proxy
                # Real logic from notebook's 'Target' implies buying now and seeing result next step
                # For simulation consistency with your provided format:
                entry_price = row['Close']
                # We assume the signal 1 means we hold for the next minute
                try:
                    exit_price = sim_df.iloc[i+1]['Close']
                    trade_profit = (num_shares * (exit_price - entry_price)) - (num_shares * cost_per_share)
                    current_balance += trade_profit
                except IndexError:
                    pass
        balances.append(current_balance)

    sim_df['Portfolio_Value'] = balances
    
    # Plot Equity Curve
    plt.figure(figsize=(12, 5))
    plt.plot(sim_df['Stock_Timestamp'], sim_df['Portfolio_Value'], color='green', linewidth=2)
    plt.axhline(initial_capital, color='red', linestyle='--', label='Initial Capital')
    plt.title(f"{symbol} GRU Equity Curve")
    plt.ylabel("Account Balance ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    equity_path = os.path.join('visualizations', f'{base_output_name}_equity.png')
    plt.savefig(equity_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Results saved to visualizations/ folder.\n")

    # Cleanup
    del model, X_train, X_val, X_test, sim_df
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
