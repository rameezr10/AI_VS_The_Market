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

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from lightning.pytorch.callbacks.progress import TQDMProgressBar

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import CrossEntropy
from torchmetrics import Accuracy
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Ignore warnings to keep logs clean
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", module="pytorch_lightning")


# =============================================================================
# CUSTOM CALLBACKS
# =============================================================================
class PrintValidationLoss(Callback):
    """Callback to print validation loss cleanly at the end of each epoch."""
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            epoch = trainer.current_epoch
            val_loss = metrics["val_loss"].item()
            logger.info(f"  Epoch {epoch:02d} | val_loss: {val_loss:.4f}")


# =============================================================================
# DATA PREPARATION
# =============================================================================
def prepare_data(symbol, data_dir='data'):
    file_path = get_latest_dataset(symbol, data_dir)
    logger.info(f"\n{'='*50}\n--- Preparing Data for {symbol} ---\n{'='*50}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Skipping {symbol}.")

    df = pd.read_csv(file_path)
    df['Stock_Timestamp'] = pd.to_datetime(df['Stock_Timestamp'], utc=True)
    df = df.sort_values('Stock_Timestamp').reset_index(drop=True)
    df['time_idx'] = df.index
    
    # Use the dynamic symbol variable
    df['symbol'] = symbol

    # Feature Engineering
    sentiment_map = {'positive': 1, 'neutral': 0, 'mixed': 0, 'negative': -1}
    df['Sentiment_Score'] = df['Sentiment'].map(sentiment_map).fillna(0)
    df['Decayed_Sentiment'] = df['Sentiment_Score'] / np.log1p(df['News_Age_Minutes'] + 1)
    df['Close_to_SMA50'] = df['Close'] / df['SMA_50']
    df['Close_to_Kalman'] = df['Close'] / df['kalman_close']

    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['Relative_Transactions'] = df['Transactions'] / df['Transactions'].rolling(window=20).mean()

    minutes_since_midnight = df['Stock_Timestamp'].dt.hour * 60 + df['Stock_Timestamp'].dt.minute
    df['Time_Sin'] = np.sin(2 * np.pi * minutes_since_midnight / 1440)
    df['Time_Cos'] = np.cos(2 * np.pi * minutes_since_midnight / 1440)

    # Target Variable
    df['Target'] = (df['Close'] > df['Close'].shift(1)).astype(int)

    df = df.dropna().reset_index(drop=True)
    df['time_idx'] = df.index 

    logger.info(f"[{symbol}] Data Prepared. Total Rows: {len(df)}")
    return df


# =============================================================================
# INDIVIDUAL PIPELINE RUNNER
# =============================================================================
def run_pipeline_for_symbol(symbol):
    def get_next_version(symbol, model_name="TFT", mode="Normal"):
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
    base_output_name = f"{symbol}_TFT_Normal_{date_str}_v{version}"

    try:
        df = prepare_data(symbol, data_dir='data')
    except FileNotFoundError as e:
        logger.info(f"❌ {e}")
        return  # Skip to the next symbol

    # 2. Configure Dataset Parameters
    max_prediction_length = 1 
    max_encoder_length = 60   

    total_len = len(df)
    train_cutoff = df["time_idx"].iloc[int(total_len * 0.7)]
    val_cutoff = df["time_idx"].iloc[int(total_len * 0.9)]

    # 3. Create TimeSeriesDataSets
    logger.info(f"\n--- Constructing TimeSeriesDataSets for {symbol} ---")
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= train_cutoff],
        time_idx="time_idx",
        target="Target",
        group_ids=["symbol"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["symbol"],
        time_varying_known_reals=["Time_Sin", "Time_Cos"],
        time_varying_unknown_reals=[
            "Close", "RSI_14", "kalman_diff", "Close_to_SMA50", "Relative_Volume", 
            "Relative_Transactions", "Decayed_Sentiment"
        ],
        target_normalizer=None, 
        scalers={
            "Close": GroupNormalizer(),
            "Relative_Volume": GroupNormalizer(),
            "Relative_Transactions": GroupNormalizer()
        },
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, df[lambda x: (x.time_idx > train_cutoff) & (x.time_idx <= val_cutoff)], predict=False)
    testing = TimeSeriesDataSet.from_dataset(training, df[lambda x: x.time_idx > val_cutoff], predict=False)

    batch_size = 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    logger.info(f"Splits: Train={len(training)}, Val={len(validation)}, Test={len(testing)}")

    # 4. Initialize Model
    logger.info(f"\n--- Initializing Model for {symbol} ---")
    acc = Accuracy(task="multiclass", num_classes=2)
    
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=128,     
        attention_head_size=4,
        dropout=0.2,
        loss=CrossEntropy(),
        logging_metrics=torch.nn.ModuleList([acc]),
        output_size=2 
    )

    # 5. Configure Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    logger.info(f"\n--- Starting Training for {symbol} using {accelerator.upper()} ---")

    checkpoint_callback = ModelCheckpoint(
        dirpath='saved_models/',
        filename=f'{base_output_name}', # Dynamic filename
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    trainer = Trainer(
        max_epochs=20,
        accelerator=accelerator,
        devices=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5), 
            checkpoint_callback,
            PrintValidationLoss()
        ],
        gradient_clip_val=0.1,
    )

    # Train the model
    trainer.fit(tft, train_dataloader, val_dataloader)
    best_model_path = checkpoint_callback.best_model_path
    logger.info(f"\n✅ Training Complete. Best {symbol} model saved at: {best_model_path}")

    # =========================================================================
    # EVALUATION & VISUALIZATION
    # =========================================================================
    logger.info(f"\n--- Generating Statistical Performance for {symbol} (10% Test Set) ---")
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    raw_predictions = best_tft.predict(test_dataloader, mode="raw", return_y=True, return_x=True)

    y_pred = raw_predictions.output.prediction.argmax(dim=-1).flatten().detach().cpu().numpy()
    y_true = raw_predictions.y[0].flatten().detach().cpu().numpy()

    # Classification Report
    cr = classification_report(y_true, y_pred, labels=[0, 1], target_names=['Down', 'Up'], zero_division=0)
    logger.info(cr)
    
    # Save Report to file dynamically
    report_path = os.path.join('visualizations', f'{base_output_name}_report.txt')
    with open(report_path, "w") as f:
        f.write(f"--- Final Statistical Performance for {symbol} (10% Test Set) ---\n")
        f.write(cr)

    # Confusion Matrix dynamically
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['Down', 'Up'], cmap='Blues')
    plt.title(f"{symbol} TFT Confusion Matrix")
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

    time_indices = raw_predictions.x['decoder_time_idx'].flatten().detach().cpu().numpy()
    sim_df = df.iloc[time_indices].copy()
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
    plt.plot(sim_df['Stock_Timestamp'], sim_df['Portfolio_Value'], color='blue', linewidth=2)
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
    del trainer, tft, best_tft, training, validation, testing, train_dataloader, val_dataloader, test_dataloader
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
