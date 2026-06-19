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

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import CrossEntropy
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score

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
# DATA PREPARATION (From TFT_baseline logic)
# =============================================================================
def prepare_data(symbol, data_dir='data'):
    file_path = get_latest_dataset(symbol, data_dir)
    logger.info(f"\n{'='*50}\n--- Preparing Data for {symbol} ---\n{'='*50}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Skipping {symbol}.")

    df = pd.read_csv(file_path)
    df['Stock_Timestamp'] = pd.to_datetime(df['Stock_Timestamp'], utc=True)
    df = df.sort_values('Stock_Timestamp').reset_index(drop=True)
    
    # Feature Selection: Baseline (Only OHLCV)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Target: 1 if the NEXT minute's candle is green (Next Close > Next Open)
    df['Target'] = (df['Close'].shift(-1) > df['Open'].shift(-1)).astype(int)

    # Time features are always known
    minutes_since_midnight = df['Stock_Timestamp'].dt.hour * 60 + df['Stock_Timestamp'].dt.minute
    df['Time_Sin'] = np.sin(2 * np.pi * minutes_since_midnight / 1440)
    df['Time_Cos'] = np.cos(2 * np.pi * minutes_since_midnight / 1440)

    # Setup specific symbol and time index
    df['symbol'] = symbol
    df = df.dropna().reset_index(drop=True)
    df['time_idx'] = df.index 

    logger.info(f"[{symbol}] Data Prepared. Total Rows: {len(df)}")
    return df, features


# =============================================================================
# INDIVIDUAL PIPELINE RUNNER
# =============================================================================
def run_pipeline_for_symbol(symbol):
    def get_next_version(symbol, model_name="TFT", mode="Baseline"):
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
    base_output_name = f"{symbol}_TFT_Baseline_{date_str}_v{version}"

    try:
        df, features = prepare_data(symbol, data_dir='data')
    except FileNotFoundError as e:
        logger.info(f"❌ {e}")
        return  # Skip to the next symbol

    # 2. Configure Dataset Parameters
    max_prediction_length = 1 
    max_encoder_length = 60   # Look back 60 minutes

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
        time_varying_unknown_reals=features, # Only OHLCV
        target_normalizer=None, 
        scalers={f: GroupNormalizer() for f in features},
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, df[lambda x: (x.time_idx > train_cutoff) & (x.time_idx <= val_cutoff)], predict=False)
    testing = TimeSeriesDataSet.from_dataset(training, df[lambda x: x.time_idx > val_cutoff], predict=False)

    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    logger.info(f"Splits: Train={len(training)}, Val={len(validation)}, Test={len(testing)}")

    # 4. Initialize Baseline Model
    logger.info(f"\n--- Initializing Model for {symbol} ---")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=32,     
        attention_head_size=4,
        dropout=0.1,
        loss=CrossEntropy(),
        output_size=2 
    )

    # 5. Configure Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    logger.info(f"\n--- Starting Training for {symbol} using {accelerator.upper()} ---")

    checkpoint_callback = ModelCheckpoint(
        dirpath='saved_models/',
        filename=f'{symbol}_tft_baseline-'+'{epoch:02d}-{val_loss:.2f}', 
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
    acc = accuracy_score(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=['Down/Flat (0)', 'Up (1)'], zero_division=0)
    logger.info(f"Test Accuracy: {acc*100:.2f}%")
    logger.info(cr)
    
    # Save Report to file dynamically
    report_path = os.path.join('visualizations', f'{base_output_name}_report.txt')
    with open(report_path, "w") as f:
        f.write(f"--- Final Statistical Performance for {symbol} (10% Test Set) ---\n")
        f.write(f"Test Accuracy: {acc*100:.2f}%\n\n")
        f.write(cr)

    # Confusion Matrix dynamically
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['Down/Flat', 'Up'], cmap='Blues')
    plt.title(f"{symbol} TFT Baseline Confusion Matrix")
    cm_path = os.path.join('visualizations', f'{base_output_name}_cm.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ Confusion Matrix saved to: {cm_path}")

    # =========================================================================
    # COMPOUNDING SIMULATION
    # =========================================================================
    logger.info(f"\n--- Running Compounding Simulation for {symbol} ---")
    INITIAL_CAPITAL = 10000.0
    current_capital = INITIAL_CAPITAL
    cost_per_share = 0.009

    time_indices = raw_predictions.x['decoder_time_idx'].flatten().detach().cpu().numpy()
    sim_df = df.iloc[time_indices].copy()
    sim_df['Signal'] = y_pred

    # Aligns with exact slippage logic (Buy T+1 Open, Sell T+1 Close)
    sim_df['Entry_Price'] = sim_df['Open'].shift(-1)
    sim_df['Exit_Price'] = sim_df['Close'].shift(-1)

    capital_history = []
    for i in range(len(sim_df)):
        row = sim_df.iloc[i]

        if row['Signal'] == 1 and not np.isnan(row['Exit_Price']) and not np.isnan(row['Entry_Price']):
            num_shares = current_capital // row['Entry_Price']
            if num_shares > 0:
                trade_profit = (num_shares * (row['Exit_Price'] - row['Entry_Price'])) - (num_shares * cost_per_share)
                current_capital += trade_profit
        capital_history.append(current_capital)

    sim_df['Account_Balance'] = capital_history
    profit = current_capital - INITIAL_CAPITAL

    logger.info("\n" + "="*50)
    logger.info(f"       {symbol} TFT BASELINE FINANCIAL PERFORMANCE")
    logger.info("="*50)
    logger.info(f"Initial Capital:         ${INITIAL_CAPITAL:,.2f}")
    logger.info(f"Net Profit/Loss:         ${profit:,.2f}")
    logger.info(f"Final Account Capital:   ${current_capital:,.2f}")
    logger.info(f"Total ROI:               {(profit/INITIAL_CAPITAL)*100:.2f}%")
    logger.info("="*50)

    # Plot Equity Curve dynamically
    plt.figure(figsize=(12, 6))
    plt.plot(sim_df['Stock_Timestamp'], sim_df['Account_Balance'], color='purple', label='Baseline Equity')
    plt.axhline(INITIAL_CAPITAL, color='red', linestyle='--', label='Initial Capital')
    plt.title(f'{symbol} Baseline TFT (OHLCV Only): Compounding Equity Curve (After Fees & Slippage)')
    plt.ylabel('Balance ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
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

def run_training(symbol, mode="Baseline"):
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    logger.info(f"Starting training for {symbol} in {mode} mode...")
    run_pipeline_for_symbol(symbol)
    logger.info(f"✅ {symbol} processed successfully!")

if __name__ == "__main__":
    run_training("MSFT")
