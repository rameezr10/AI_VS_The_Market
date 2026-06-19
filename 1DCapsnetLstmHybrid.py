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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Ignore warnings to keep logs clean
warnings.filterwarnings("ignore")

# =============================================================================
# CUSTOM MODEL (Placeholder for your CapsNet-LSTM architecture)
# =============================================================================
# Please replace the below dummy class with your actual CapsuleNetworkLSTM model 
# exactly as it is in your .ipynb file.

class CapsuleNetworkLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CapsuleNetworkLSTM, self).__init__()
        # TODO: Paste your model layers here
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # TODO: Paste your forward pass logic here
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out


# =============================================================================
# DATA PREPARATION
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
    
    # Ensure Timestamp parsing
    if 'Stock_Timestamp' in df.columns:
        df['Stock_Timestamp'] = pd.to_datetime(df['Stock_Timestamp'], utc=True)
        df = df.sort_values('Stock_Timestamp').reset_index(drop=True)
    
    # Feature Engineering (Preserved from 1Dcapsnetlstmhybrid.ipynb)
    sentiment_map = {'positive': 1, 'neutral': 0, 'mixed': 0, 'negative': -1}
    df['Sentiment_Score'] = df['Sentiment'].map(sentiment_map).fillna(0)
    
    if 'News_Age_Minutes' in df.columns:
        df['Decayed_Sentiment'] = df['Sentiment_Score'] / np.log1p(df['News_Age_Minutes'] + 1)
    else:
        df['Decayed_Sentiment'] = df['Sentiment_Score']

    features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'RSI_14', 
        'kalman_close', 'kalman_diff', 'Decayed_Sentiment'
    ]

    # Target Logic
    df['Target'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)
    
    # Drop rows with NaN in features or target
    df_clean = df.dropna(subset=features + ['Target']).copy().reset_index(drop=True)

    logger.info(f"[{symbol}] Data Prepared. Total Rows: {len(df_clean)}")
    return df_clean, features


# =============================================================================
# INDIVIDUAL PIPELINE RUNNER
# =============================================================================
def run_pipeline_for_symbol(symbol):
    def get_next_version(symbol, model_name="1DCapsnetLstmHybrid", mode="Normal"):
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
    base_output_name = f"{symbol}_1DCapsnetLstmHybrid_Normal_{date_str}_v{version}"

    try:
        df, features = prepare_data(symbol, data_dir='data')
    except FileNotFoundError as e:
        logger.info(f"❌ {e}")
        return  # Skip to the next symbol

    # 1. Create Sequences
    logger.info(f"\n--- Constructing Train/Val/Test Splits for {symbol} ---")
    time_steps = 10
    X_seq, y_seq = create_sequences(df[features].values, df['Target'].values, time_steps)
    
    # Store indices for financial tracking/compounding simulation later
    # The first target is mapped to index `time_steps`
    valid_indices = df.index[time_steps:].values 

    # 2. Split Data (70% Train, 20% Val, 10% Test)
    train_size = int(len(X_seq) * 0.7)
    val_size = int(len(X_seq) * 0.2)

    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_val, y_val = X_seq[train_size:train_size+val_size], y_seq[train_size:train_size+val_size]
    X_test, y_test = X_seq[train_size+val_size:], y_seq[train_size+val_size:]
    test_indices = valid_indices[train_size+val_size:]

    # 3. Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, len(features))).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, len(features))).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, len(features))).reshape(X_test.shape)

    # 4. DataLoaders
    batch_size = 64
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=batch_size, shuffle=False)

    logger.info(f"Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 5. Initialize Model
    logger.info(f"\n--- Initializing Model for {symbol} ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CapsuleNetworkLSTM(input_dim=len(features), hidden_dim=64, output_dim=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logger.info(f"\n--- Starting Training for {symbol} using {str(device).upper()} ---")
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join('saved_models', f'{base_output_name}.pth')

    # 6. Training Loop
    for epoch in range(1, 51):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        logger.info(f"  Epoch [{epoch:02d}/50] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    logger.info(f"\n✅ Training Complete. Best {symbol} model saved at: {best_model_path}")

    # =========================================================================
    # EVALUATION & VISUALIZATION
    # =========================================================================
    logger.info(f"\n--- Generating Statistical Performance for {symbol} (Test Set) ---")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    y_pred_list = []
    y_true_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            y_pred_list.extend(predicted.cpu().numpy())
            y_true_list.extend(y_batch.numpy())

    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true_list)

    # Classification Report
    cr = classification_report(y_true, y_pred, labels=[0, 1], target_names=['Down', 'Up'], zero_division=0)
    logger.info(cr)
    
    # Save Report to file dynamically
    report_path = os.path.join('visualizations', f'{base_output_name}_report.txt')
    with open(report_path, "w") as f:
        f.write(f"--- Final Statistical Performance for {symbol} (Test Set) ---\n")
        f.write(cr)

    # Confusion Matrix dynamically
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['Down', 'Up'], cmap='Blues')
    plt.title(f"{symbol} CapsNet-LSTM Confusion Matrix")
    cm_path = os.path.join('visualizations', f'{base_output_name}_cm.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Confusion Matrix saved to: {cm_path}")

    # =========================================================================
    # COMPOUNDING SIMULATION (Using the Test Set)
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
    logger.info(f"[{symbol}] Total Return:    {((current_balance - initial_capital)/initial_capital)*100:,.2f}%")

    # Plot Equity Curve dynamically
    plt.figure(figsize=(12, 5))
    if 'Stock_Timestamp' in sim_df.columns:
        plt.plot(sim_df['Stock_Timestamp'], sim_df['Portfolio_Value'], color='blue', linewidth=2)
    else:
        plt.plot(sim_df.index, sim_df['Portfolio_Value'], color='blue', linewidth=2)
        
    plt.axhline(initial_capital, color='red', linestyle='--', label='Initial Capital')
    plt.title(f"{symbol} Equity Curve: Full Reinvestment (Compounding)")
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
    del model, train_loader, val_loader, test_loader, X_train, X_val, X_test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
