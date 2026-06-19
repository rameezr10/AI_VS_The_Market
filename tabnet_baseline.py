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
import torch
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Ignore warnings to keep logs clean
warnings.filterwarnings("ignore")

# =============================================================================
# DATA PREPARATION (LOGIC FROM TABNET_BASELINE.IPYNB)
# =============================================================================
def prepare_data(symbol, data_dir='data'):
    file_path = get_latest_dataset(symbol, data_dir)
    logger.info(f"\n{'='*50}\n--- Preparing Data for {symbol} ---\n{'='*50}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Skipping {symbol}.")

    df = pd.read_csv(file_path)
    df['Stock_Timestamp'] = pd.to_datetime(df['Stock_Timestamp'], utc=True)
    df = df.sort_values('Stock_Timestamp').reset_index(drop=True)

    # Baseline Feature Selection (OHLCV Only)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Target: 1 if the NEXT minute's candle is green (Next Close > Next Open)
    df['Target'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)
    
    # Drop the last row as it won't have a target
    df = df.dropna().reset_index(drop=True)

    logger.info(f"[{symbol}] Data Prepared. Total Rows: {len(df)}")
    return df, feature_cols


# =============================================================================
# INDIVIDUAL PIPELINE RUNNER
# =============================================================================
def run_pipeline_for_symbol(symbol):
    def get_next_version(symbol, model_name="tabnet", mode="Baseline"):
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
    base_output_name = f"{symbol}_tabnet_Baseline_{date_str}_v{version}"

    try:
        df, feature_cols = prepare_data(symbol, data_dir='data')
    except FileNotFoundError as e:
        logger.info(f"❌ {e}")
        return

    # --- 1. CHRONOLOGICAL SPLIT (70/20/10) ---
    n = len(df)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.90) 

    X = df[feature_cols].values
    y = df['Target'].values

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    # --- 2. SCALING (FIT ONLY ON TRAIN) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # --- 3. TABNET MODEL DEFINITION ---
    logger.info(f"\n--- Initializing TabNet for {symbol} ---")
    tabnet_model = TabNetClassifier(
        n_d=16, n_a=16, n_steps=3,
        gamma=1.3, n_independent=2, n_shared=2,
        lambda_sparse=1e-3, optimizer_params=dict(lr=2e-2),
        mask_type='entmax' 
    )

    # --- 4. TRAINING ---
    logger.info(f"--- Starting Training for {symbol} ---")
    tabnet_model.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['valid'],
        eval_metric=['accuracy'],
        max_epochs=50, 
        patience=10,
        batch_size=1024, 
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    # Save Model
    model_save_path = os.path.join('saved_models', f'{symbol}_tabnet_baseline')
    tabnet_model.save_model(model_save_path)
    logger.info(f"✅ Model saved at: {model_save_path}")

    # =========================================================================
    # EVALUATION & VISUALIZATION
    # =========================================================================
    logger.info(f"\n--- Generating Statistical Performance for {symbol} ---")
    y_pred_probs = tabnet_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_probs > 0.50).astype(int)

    # Classification Report
    cr = classification_report(y_test, y_pred, target_names=['Down/Flat (0)', 'Up (1)'])
    logger.info(cr)
    
    # Save Report
    report_path = os.path.join('visualizations', f'{base_output_name}_report.txt')
    with open(report_path, "w") as f:
        f.write(f"--- TabNet Baseline Performance for {symbol} ---\n")
        f.write(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n\n")
        f.write(cr)

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Down', 'Up'], cmap='Blues')
    plt.title(f"{symbol} TabNet Confusion Matrix")
    cm_path = os.path.join('visualizations', f'{base_output_name}_cm.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FINANCIAL PERFORMANCE (COMPOUNDING SIMULATION)
    # =========================================================================
    logger.info(f"--- Running Financial Simulation for {symbol} ---")
    INITIAL_CAPITAL = 10000.0
    COST_PER_SHARE = 0.009
    current_capital = INITIAL_CAPITAL
    
    # We use the test portion of the original dataframe
    df_test = df.iloc[val_idx:].copy()
    df_test['Prediction'] = y_pred
    
    # Logic: If Prediction is 1, buy at Open of the NEXT bar and sell at Close of that same NEXT bar
    # Because Target was (df['Close'].shift(-1) > df['Open'].shift(-1))
    
    capital_history = []
    
    # Note: Shifted values to align with the execution logic
    # df_test['Target'] corresponds to the candle at index i+1
    # So we trade the candle at index i+1 using prediction at index i
    
    next_opens = df['Open'].shift(-1).iloc[val_idx:].values
    next_closes = df['Close'].shift(-1).iloc[val_idx:].values

    for i in range(len(df_test)):
        signal = y_pred[i]
        entry_price = next_opens[i]
        exit_price = next_closes[i]

        if signal == 1 and not np.isnan(exit_price):
            num_shares = current_capital // entry_price
            if num_shares > 0:
                trade_profit = (num_shares * (exit_price - entry_price)) - (num_shares * COST_PER_SHARE)
                current_capital += trade_profit
        
        capital_history.append(current_capital)

    df_test['Account_Balance'] = capital_history
    profit = current_capital - INITIAL_CAPITAL
    roi = (profit / INITIAL_CAPITAL) * 100

    logger.info(f"[{symbol}] Final Balance: ${current_capital:,.2f} (ROI: {roi:.2f}%)")

    # Plot Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(df_test['Stock_Timestamp'], df_test['Account_Balance'], color='blue', label='Equity Curve')
    plt.axhline(INITIAL_CAPITAL, color='red', linestyle='--', label='Initial Capital')
    plt.title(f'TabNet Baseline ({symbol}): Financial Growth')
    plt.ylabel('Balance ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    equity_path = os.path.join('visualizations', f'{base_output_name}_equity.png')
    plt.savefig(equity_path, dpi=300, bbox_inches='tight')
    plt.close()

    
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

    # Clean up memory
    del tabnet_model, X_train, X_val, X_test, df, df_test
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
