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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier

# Ignore warnings to keep logs clean
warnings.filterwarnings("ignore")

# =============================================================================
# DATA PREPARATION (LOGIC FROM TABNET.IPYNB)
# =============================================================================
def prepare_data(symbol, data_dir='data'):
    file_path = get_latest_dataset(symbol, data_dir)
    logger.info(f"\n{'='*50}\n--- Preparing Data for {symbol} ---\n{'='*50}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Skipping {symbol}.")

    df = pd.read_csv(file_path)
    
    # Preprocessing Logic from Notebook
    sentiment_map = {'positive': 1, 'neutral': 0, 'mixed': 0, 'negative': -1}
    df['Sentiment_Score'] = df['Sentiment'].map(sentiment_map).fillna(0)
    df['Decayed_Sentiment'] = df['Sentiment_Score'] / np.log1p(df['News_Age_Minutes'] + 1)

    # Target Logic: Next Open > Current Close
    df['Target'] = (df['Open'].shift(-1) > df['Close']).astype(int)

    # Features used in TabNet notebook
    features = [
        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 
        'EMA_10', 'EMA_50', 'rolling_high_20',
        'kalman_close', 'kalman_diff', 'Decayed_Sentiment'
    ]

    # Clean and Drop NaNs
    df = df.dropna(subset=features + ['Target']).reset_index(drop=True)
    
    logger.info(f"[{symbol}] Data Prepared. Total Rows: {len(df)}")
    return df, features

# =============================================================================
# INDIVIDUAL PIPELINE RUNNER
# =============================================================================
def run_pipeline_for_symbol(symbol):
    def get_next_version(symbol, model_name="tabnet", mode="Normal"):
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
    base_output_name = f"{symbol}_tabnet_Normal_{date_str}_v{version}"

    try:
        df, features = prepare_data(symbol, data_dir='data')
    except FileNotFoundError as e:
        logger.info(f"❌ {e}")
        return

    # 1. Data Splitting (70% Train, 20% Val, 10% Test)
    X = df[features].values
    y = df['Target'].values

    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.2)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    # For simulation tracking
    test_df = df.iloc[train_size+val_size:].copy()

    # 2. Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 3. Initialize TabNet Model
    clf = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='sparsemax' 
    )

    # 4. Training
    logger.info(f"--- Training TabNet for {symbol} ---")
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_name=['train', 'valid'],
        eval_metric=['accuracy'],
        max_epochs=50, 
        patience=10, # Enabled patience for better convergence
        batch_size=1024, 
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    # 5. Save Model
    model_path = os.path.join("saved_models", f"{symbol}_tabnet_model")
    clf.save_model(model_path)
    logger.info(f"✅ Model saved at: {model_path}")

    # =========================================================================
    # EVALUATION & VISUALIZATION
    # =========================================================================
    logger.info(f"\n--- Evaluation for {symbol} ---")
    # Confidence Thresholding (60% from Notebook logic)
    CONFIDENCE_THRESHOLD = 0.60
    probs = clf.predict_proba(X_test)[:, 1]
    y_pred = (probs >= CONFIDENCE_THRESHOLD).astype(int)
    
    # Classification Report
    cr = classification_report(y_test, y_pred, labels=[0, 1], target_names=['Down', 'Up'], zero_division=0)
    logger.info(cr)
    
    report_path = os.path.join('visualizations', f'{base_output_name}_report.txt')
    with open(report_path, "w") as f:
        f.write(f"--- TabNet Performance for {symbol} ---\n")
        f.write(cr)

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Down', 'Up'], cmap='Greens')
    plt.title(f"{symbol} TabNet Confusion Matrix")
    cm_path = os.path.join('visualizations', f'{base_output_name}_cm.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # =========================================================================
    # COMPOUNDING SIMULATION
    # =========================================================================
    logger.info(f"--- Running Simulation for {symbol} ---")
    initial_capital = 10000.0
    current_balance = initial_capital
    cost_per_share = 0.009  # From standard format

    test_df['Signal'] = y_pred
    test_df['Entry_Price'] = test_df['Close']         # Buy at the very end of today
    test_df['Exit_Price'] = test_df['Open'].shift(-1)

    balances = []
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        # Only trade if model is confident (Signal == 1)
        if row['Signal'] == 1 and not np.isnan(row['Exit_Price']):
            num_shares = current_balance // row['Entry_Price']
            if num_shares > 0:
                trade_profit = (num_shares * (row['Exit_Price'] - row['Entry_Price'])) - (num_shares * cost_per_share)
                current_balance += trade_profit
        balances.append(current_balance)

    test_df['Portfolio_Value'] = balances

    # Plot Equity Curve
    plt.figure(figsize=(12, 5))
    plt.plot(test_df['Portfolio_Value'], color='tab:blue', label='Equity Curve')
    plt.axhline(initial_capital, color='red', linestyle='--', label='Initial Capital')
    plt.title(f"{symbol} TabNet Equity Curve (Threshold: {CONFIDENCE_THRESHOLD*100}%)")
    plt.ylabel("Account Balance ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    equity_path = os.path.join('visualizations', f'{base_output_name}_equity.png')
    plt.savefig(equity_path, dpi=300)
    plt.close()
    logger.info(f"✅ Simulation Complete. Final Balance: ${current_balance:,.2f}\n")

    # Memory cleanup
    del clf, X_train, X_val, X_test, df
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
