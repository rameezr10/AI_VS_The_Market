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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Ignore warnings to keep logs clean
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Reduce TF logging

# =============================================================================
# DATA PREPARATION (Baseline OHLCV Logic)
# =============================================================================
def prepare_data(symbol, data_dir='data', window_size=20):
    file_path = get_latest_dataset(symbol, data_dir)
    logger.info(f"\n{'='*50}\n--- Preparing Data for {symbol} ---\n{'='*50}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Skipping {symbol}.")

    df = pd.read_csv(file_path)
    df['Stock_Timestamp'] = pd.to_datetime(df['Stock_Timestamp'], utc=True)
    df = df.sort_values('Stock_Timestamp').reset_index(drop=True)

    # --- Feature Selection (Baseline: OHLCV only) ---
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Target: 1 if the NEXT minute's candle is green (Next Close > Next Open)
    df['Target'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)

    # Clean and prepare model data
    model_df = df[feature_cols + ['Target']].dropna()
    raw_test_df = df.iloc[model_df.index].copy() # Keep for financial simulation

    # --- Chronological Split (70/20/10) ---
    n = len(model_df)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.90)

    train_data = model_df.iloc[:train_idx]
    val_data = model_df.iloc[train_idx:val_idx]
    test_data = model_df.iloc[val_idx:]

    # --- Scaling (Fit only on Train) ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data[feature_cols])
    val_scaled = scaler.transform(val_data[feature_cols])
    test_scaled = scaler.transform(test_data[feature_cols])

    # Combine scaled features with targets
    train_final = np.column_stack((train_scaled, train_data['Target'].values))
    val_final = np.column_stack((val_scaled, val_data['Target'].values))
    test_final = np.column_stack((test_scaled, test_data['Target'].values))

    # --- Windowing / Sequence Creation ---
    def create_sequences(data, window):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i, :-1])
            y.append(data[i, -1])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_final, window_size)
    X_val, y_val = create_sequences(val_final, window_size)
    X_test, y_test = create_sequences(test_final, window_size)

    # Financial test set alignment (rows corresponding to X_test predictions)
    # y_test[i] is the target for test_data row (window_size + i)
    sim_df = test_data.iloc[window_size:].copy()
    sim_df['Stock_Timestamp'] = raw_test_df.iloc[val_idx + window_size:]['Stock_Timestamp'].values

    logger.info(f"[{symbol}] Data Prepared. Train sequences: {len(X_train)}, Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, sim_df, len(feature_cols)


# =============================================================================
# INDIVIDUAL PIPELINE RUNNER
# =============================================================================
def run_pipeline_for_symbol(symbol):
    def get_next_version(symbol, model_name="gru", mode="Baseline"):
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
    base_output_name = f"{symbol}_gru_Baseline_{date_str}_v{version}"

    WINDOW_SIZE = 20
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, sim_df, input_dim = prepare_data(symbol, window_size=WINDOW_SIZE)
    except FileNotFoundError as e:
        logger.info(f"❌ {e}")
        return

    # 1. Initialize Model (Baseline GRU Architecture)
    logger.info(f"\n--- Initializing Baseline GRU Model for {symbol} ---")
    model = Sequential([
        Input(shape=(WINDOW_SIZE, input_dim)),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # 2. Training
    logger.info(f"--- Starting Training for {symbol} ---")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )

    # Save Model
    model_path = os.path.join('saved_models', f'{symbol}_gru_baseline.h5')
    model.save(model_path)
    logger.info(f"✅ Model saved at: {model_path}")

    # 3. Evaluation & Visualization
    logger.info(f"\n--- Generating Statistical Performance for {symbol} ---")
    probs = model.predict(X_test, verbose=0)
    y_pred = (probs > 0.5).astype(int).flatten()

    # Classification Report
    cr = classification_report(y_test, y_pred, target_names=['Down', 'Up'], zero_division=0)
    logger.info(cr)
    report_path = os.path.join('visualizations', f'{base_output_name}_report.txt')
    with open(report_path, "w") as f:
        f.write(f"--- GRU Baseline Performance for {symbol} ---\n")
        f.write(cr)

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Down', 'Up'], cmap='Greys')
    plt.title(f"{symbol} GRU Baseline Confusion Matrix")
    cm_path = os.path.join('visualizations', f'{base_output_name}_cm.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Compounding Simulation (Notebook Logic: Entry at Open, Exit at Close)
    logger.info(f"--- Running Compounding Simulation for {symbol} ---")
    initial_capital = 10000.0
    current_balance = initial_capital
    slippage = 0.009 # From standard template

    sim_df['Signal'] = y_pred
    balances = []
    
    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        if row['Signal'] == 1:
            # Entry at Open, Exit at Close of the SAME minute (per Target logic)
            num_shares = current_balance // row['Open']
            if num_shares > 0:
                trade_profit = (num_shares * (row['Close'] - row['Open'])) - (num_shares * slippage)
                current_balance += trade_profit
        balances.append(current_balance)

    sim_df['Portfolio_Value'] = balances
    logger.info(f"[{symbol}] Final Balance: ${current_balance:,.2f} | ROI: {((current_balance - initial_capital)/initial_capital)*100:.2f}%")

    # Plot Equity Curve
    plt.figure(figsize=(12, 5))
    plt.plot(sim_df['Stock_Timestamp'], sim_df['Portfolio_Value'], color='black', linewidth=1.5)
    plt.axhline(initial_capital, color='red', linestyle='--', label='Initial Capital')
    plt.title(f"{symbol} GRU Baseline Equity Curve")
    plt.ylabel("Account Balance ($)")
    plt.grid(alpha=0.3)
    
    equity_path = os.path.join('visualizations', f'{base_output_name}_equity.png')
    plt.savefig(equity_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Cleanup
    del model, X_train, X_val, X_test, sim_df
    tf.keras.backend.clear_session()
    gc.collect()


# =============================================================================
# MAIN LOOP
# =============================================================================

def run_training(symbol, mode="Baseline"):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    logger.info(f"Starting training for {symbol} in {mode} mode...")
    run_pipeline_for_symbol(symbol)
    logger.info(f"✅ {symbol} processed successfully!")

if __name__ == "__main__":
    run_training("MSFT")
