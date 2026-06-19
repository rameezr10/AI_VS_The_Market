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
import copy
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

# Ignore warnings to keep logs clean
warnings.filterwarnings("ignore")

# =============================================================================
# CUSTOM MODEL (1D CapsNet-LSTM Baseline)
# =============================================================================
def squash(x):
    s_sq_norm = torch.sum(x**2, dim=-1, keepdim=True)
    return (s_sq_norm / (1 + s_sq_norm)) * (x / torch.sqrt(s_sq_norm + 1e-9))

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules=8, num_route_nodes=10, in_channels=64, out_channels=16):
        super().__init__()
        # Dynamic routing parameters
        self.W = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        
    def forward(self, x):
        x = x[:, None, :, :, None]
        u_hat = torch.matmul(self.W[None, ...].transpose(-1, -2), x).squeeze(-1)
        b_ij = torch.zeros(u_hat.size(0), 8, 10).to(x.device) # Routing logits
        
        for i in range(3): # 3 iterations of dynamic routing
            c_ij = F.softmax(b_ij, dim=1)
            v_j = squash((c_ij[:, :, :, None] * u_hat).sum(dim=2))
            if i < 2: b_ij = b_ij + (u_hat * v_j[:, :, None, :]).sum(dim=-1)
        return v_j

class HybridCapsNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.caps = CapsuleLayer(8, 10, 64, 16)
        self.lstm = nn.LSTM(16, 32, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(32, 2)
        
    def forward(self, x):
        # 1D Conv -> Primary Capsules -> LSTM Temporal Integration
        x = F.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        v_j = self.caps(x)
        _, (h_n, _) = self.lstm(v_j)
        return F.softmax(self.fc(h_n[-1]), dim=1)


# =============================================================================
# DATA PREPARATION (Baseline OHLCV)
# =============================================================================
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def prepare_data(symbol, data_dir='data'):
    file_path = get_latest_dataset(symbol, data_dir)
    logger.info(f"\n{'='*50}\n--- Preparing Data for {symbol} ---\n{'='*50}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Skipping {symbol}.")

    df = pd.read_csv(file_path)
    
    if 'Stock_Timestamp' in df.columns:
        df['Stock_Timestamp'] = pd.to_datetime(df['Stock_Timestamp'], utc=True)
        df = df.sort_values('Stock_Timestamp').reset_index(drop=True)
    
    # Baseline Features
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Target Logic: 1 if Next Close > Next Open (Matches T+1 Open entry execution)
    df['Target'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)
    
    # Drop rows with NaN
    df_clean = df.dropna(subset=feature_cols + ['Target']).copy().reset_index(drop=True)

    logger.info(f"[{symbol}] Baseline Data Prepared. Total Rows: {len(df_clean)}")
    return df_clean, feature_cols


# =============================================================================
# INDIVIDUAL PIPELINE RUNNER
# =============================================================================
def run_pipeline_for_symbol(symbol):
    def get_next_version(symbol, model_name="1DCapsnetLstmHybrid", mode="Baseline"):
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
    base_output_name = f"{symbol}_1DCapsnetLstmHybrid_Baseline_{date_str}_v{version}"

    try:
        df, feature_cols = prepare_data(symbol, data_dir='data')
    except FileNotFoundError as e:
        logger.info(f"❌ {e}")
        return  # Skip to the next symbol

    # 1. Create Sequences
    logger.info(f"\n--- Constructing Train/Val/Test Splits for {symbol} ---")
    time_steps = 10
    X_seq, y_seq = create_sequences(df[feature_cols].values, df['Target'].values, time_steps)
    
    # Track indices for accurate plotting/compounding simulation later
    valid_indices = df.index[time_steps:].values 

    # 2. Chronological Split (70/20/10)
    train_idx = int(len(X_seq) * 0.70)
    val_idx = int(len(X_seq) * 0.90)

    X_train_raw, y_train = X_seq[:train_idx], y_seq[:train_idx]
    X_val_raw, y_val = X_seq[train_idx:val_idx], y_seq[train_idx:val_idx]
    X_test_raw, y_test = X_seq[val_idx:], y_seq[val_idx:]
    test_indices = valid_indices[val_idx:]

    # 3. Scaling (Fit ONLY on Train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw.reshape(-1, len(feature_cols))).reshape(X_train_raw.shape)
    X_val = scaler.transform(X_val_raw.reshape(-1, len(feature_cols))).reshape(X_val_raw.shape)
    X_test = scaler.transform(X_test_raw.reshape(-1, len(feature_cols))).reshape(X_test_raw.shape)

    # 4. DataLoaders
    batch_size = 64
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=batch_size, shuffle=False)

    logger.info(f"Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 5. Initialize Model
    logger.info(f"\n--- Initializing Baseline HybridCapsNet for {symbol} ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridCapsNet(input_dim=len(feature_cols)).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    EPOCHS = 30
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_path = os.path.join('saved_models', f'{base_output_name}.pth')

    logger.info(f"\n--- Starting Training for {symbol} using {str(device).upper()} ---")

    # 6. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for v_data, v_target in val_loader:
                v_data, v_target = v_data.to(device), v_target.to(device)
                v_output = model(v_data)
                preds = torch.argmax(v_output, dim=1)
                correct += (preds == v_target).sum().item()
        
        val_acc = correct / len(X_val)
        logger.info(f"  Epoch [{epoch+1:02d}/{EPOCHS}] | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, best_model_path)

    logger.info(f"\n✅ Training Complete. Best {symbol} model saved at: {best_model_path}")

    # =========================================================================
    # EVALUATION & VISUALIZATION
    # =========================================================================
    logger.info(f"\n--- Generating Statistical Performance for {symbol} (10% Test Set) ---")
    model.load_state_dict(best_model_wts)
    model.eval()

    y_pred_probs = []
    with torch.no_grad():
        for t_data, _ in test_loader:
            t_data = t_data.to(device)
            # Using soft probability of class '1' > 0.5 as defined in baseline
            y_pred_probs.extend(model(t_data).cpu().numpy()[:, 1])

    y_pred = (np.array(y_pred_probs) > 0.50).astype(int)

    # Classification Report
    cr = classification_report(y_test, y_pred, target_names=['Down/Flat', 'Up'], zero_division=0)
    logger.info(f"Test Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    logger.info(cr)
    
    # Save Report
    report_path = os.path.join('visualizations', f'{base_output_name}_report.txt')
    with open(report_path, "w") as f:
        f.write(f"--- CAPSNET-LSTM BASELINE STATISTICAL PERFORMANCE for {symbol} ---\n")
        f.write(f"Test Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%\n\n")
        f.write(cr)

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Down/Flat', 'Up'], cmap='Blues')
    plt.title(f"{symbol} Baseline CapsNet-LSTM Confusion Matrix")
    cm_path = os.path.join('visualizations', f'{base_output_name}_cm.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Confusion Matrix saved to: {cm_path}")

    # =========================================================================
    # COMPOUNDING SIMULATION
    # =========================================================================
    logger.info(f"\n--- Running Compounding Simulation for {symbol} ---")
    initial_capital = 10000.0
    current_balance = initial_capital
    cost_per_share = 0.009

    # Re-slice the actual dataframe using test_indices
    sim_df = df.iloc[test_indices].copy()
    sim_df['Signal'] = y_pred

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
    logger.info(f"[{symbol}] Total ROI:       {((current_balance - initial_capital)/initial_capital)*100:,.2f}%")

    # Plot Equity Curve
    plt.figure(figsize=(12, 6))
    if 'Stock_Timestamp' in sim_df.columns:
        plt.plot(sim_df['Stock_Timestamp'], sim_df['Portfolio_Value'], color='navy', label='Baseline Equity', linewidth=2)
    else:
        plt.plot(sim_df.index, sim_df['Portfolio_Value'], color='navy', label='Baseline Equity', linewidth=2)
        
    plt.axhline(initial_capital, color='red', linestyle='--', label='Initial Capital')
    plt.title(f"{symbol} Baseline Hybrid CapsNet: Financial Performance")
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

    # Clean up memory
    del model, train_loader, val_loader, test_loader, X_train, X_val, X_test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
