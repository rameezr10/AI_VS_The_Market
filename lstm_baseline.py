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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Ignore warnings to keep logs clean
warnings.filterwarnings("ignore")

# =============================================================================
# MODEL ARCHITECTURE (From lstm_baseline.ipynb)
# =============================================================================
class BaselineLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super(BaselineLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        logits = self.fc(last_time_step_out)
        return F.softmax(logits, dim=1)

# =============================================================================
# DATA PREPARATION (From lstm_baseline.ipynb logic)
# =============================================================================
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def prepare_data(symbol, data_dir='data', lookback=10):
    file_path = get_latest_dataset(symbol, data_dir)
    logger.info(f"\n{'='*50}\n--- Preparing Data for {symbol} ---\n{'='*50}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Skipping {symbol}.")

    df = pd.read_csv(file_path)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Target logic from your notebook
    df['Target'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)
    df = df.dropna(subset=features + ['Target']).reset_index(drop=True)
    
    X_raw = df[features].values
    y_raw = df['Target'].values

    X_seq, y_seq = create_sequences(X_raw, y_raw, lookback)
    
    train_size = int(len(X_seq) * 0.7)
    val_size = int(len(X_seq) * 0.2)

    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_val, y_val = X_seq[train_size:train_size+val_size], y_seq[train_size:train_size+val_size]
    X_test, y_test = X_seq[train_size+val_size:], y_seq[train_size+val_size:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, len(features))).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, len(features))).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, len(features))).reshape(X_test.shape)
    
    test_df = df.iloc[train_size + val_size + lookback:].copy()

    return {
        'train': (X_train_scaled, y_train),
        'val': (X_val_scaled, y_val),
        'test': (X_test_scaled, y_test),
        'test_df': test_df,
        'feature_count': len(features)
    }

# =============================================================================
# INDIVIDUAL PIPELINE RUNNER
# =============================================================================
def run_pipeline_for_symbol(symbol):
    def get_next_version(symbol, model_name="lstm", mode="Baseline"):
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
    base_output_name = f"{symbol}_lstm_Baseline_{date_str}_v{version}"

    try:
        data_package = prepare_data(symbol, data_dir='data')
    except FileNotFoundError as e:
        logger.info(f"❌ {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Loaders
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(data_package['train'][0]), 
                                            torch.LongTensor(data_package['train'][1])), 
                              batch_size=64, shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(data_package['val'][0]), 
                                          torch.LongTensor(data_package['val'][1])), 
                            batch_size=64, shuffle=False)

    # 2. Model Initialization
    model = BaselineLSTM(input_dim=data_package['feature_count']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 3. Training (Notebook Logic)
    epochs = 15
    logger.info(f"\n--- Training {symbol} on {str(device).upper()} ---")
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            criterion(model(data), target).backward()
            optimizer.step()

    # 4. Save Model
    model_path = os.path.join('saved_models', f'{symbol}_lstm_baseline.pth')
    torch.save(model.state_dict(), model_path)

    # 5. EVALUATION & CONFUSION MATRIX
    logger.info(f"\n--- Generating Performance Results for {symbol} ---")
    model.eval()
    X_test_tensor = torch.FloatTensor(data_package['test'][0]).to(device)
    y_test = data_package['test'][1]
    
    with torch.no_grad():
        probs = model(X_test_tensor).cpu().numpy()[:, 1]
    
    y_pred = (probs > 0.50).astype(int)

    # Classification Report
    cr = classification_report(y_test, y_pred, target_names=['Down', 'Up'], zero_division=0)
    logger.info(cr)
    report_path = os.path.join('visualizations', f'{base_output_name}_report.txt')
    with open(report_path, "w") as f:
        f.write(cr)

    # --- ADDED: Confusion Matrix Visualization ---
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Down', 'Up'], cmap='Blues')
    plt.title(f"{symbol} LSTM Confusion Matrix")
    cm_path = os.path.join('visualizations', f'{base_output_name}_cm.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Confusion Matrix saved to: {cm_path}")

    # 6. COMPOUNDING SIMULATION
    logger.info(f"\n--- Running Financial Simulation for {symbol} ---")
    current_capital = 10000.0
    cost_per_share = 0.009
    df_test = data_package['test_df']
    df_test['Prediction'] = y_pred
    
    capital_history = []
    for i in range(len(df_test)):
        row = df_test.iloc[i]
        if row['Prediction'] == 1:
            num_shares = current_capital // row['Open']
            if num_shares > 0:
                current_capital += (num_shares * (row['Close'] - row['Open'])) - (num_shares * cost_per_share)
        capital_history.append(current_capital)

    df_test['Account_Balance'] = capital_history

    # Equity Curve Plot
    plt.figure(figsize=(12, 5))
    plt.plot(df_test['Account_Balance'], color='blue', linewidth=2)
    plt.axhline(10000.0, color='red', linestyle='--', label='Initial Capital')
    plt.title(f"{symbol} LSTM Equity Curve")
    plt.ylabel("Account Balance ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    equity_path = os.path.join('visualizations', f'{base_output_name}_equity.png')
    plt.savefig(equity_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Equity Curve saved to: {equity_path}\n")

    # Cleanup
    del model, train_loader, val_loader, X_test_tensor, data_package
    if torch.cuda.is_available(): torch.cuda.empty_cache()
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
