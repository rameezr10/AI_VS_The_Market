import os
import re
import glob
import json
import plotly
import plotly.graph_objects as go
import pandas as pd
import logging
from importlib import import_module

logger = logging.getLogger(__name__)

# Map the UI model names to their underlying python script names (Normal, Baseline)
MODEL_MAP = {
    'TFT': ('TFT', 'TFT_baseline'),
    'GRU': ('gru', 'gru_baseline'),
    'LSTM': ('lstm', 'lstm_baseline'),
    'TabNet': ('tabnet', 'tabnet_baseline'),
    'CapsNetLSTM': ('1DCapsnetLstmHybrid', '1DCapsnetLstmHybrid_baseline')
}

def trigger_collection(symbol):
    """
    Triggers a full-feature data collection for the given symbol.
    """
    try:
        from data_collection import run_collection
        import data_visualizations
        
        logger.info(f"Triggering data collection for {symbol}")
        # run_collection saves the file and handles the datetime logic
        output_file = run_collection(symbol)
        
        # After collection, automatically generate the technical dashboard
        logger.info(f"Generating technical dashboard for {symbol}")
        data_visualizations.generate_for_ticker(symbol)
        
        return output_file
    except Exception as e:
        logger.error(f"Error in trigger_collection for {symbol}: {e}")
        return None

def trigger_training(symbol, model_type, mode):
    """
    Triggers the training pipeline for a specific model and mode.
    """
    logger.info(f"Triggering training for {symbol} using {model_type} in {mode} mode")
    try:
        if model_type not in MODEL_MAP:
            raise ValueError(f"Unknown model_type {model_type}")
            
        module_name = MODEL_MAP[model_type][0] if mode == "Normal" else MODEL_MAP[model_type][1]
        model_module = import_module(module_name)
        
        # This will load the latest dataset, run training, and save a versioned checkpoint
        model_module.run_training(symbol, mode)
        return True
    except Exception as e:
        logger.error(f"Error in trigger_training for {symbol}: {e}")
        return False

def get_dataset_chunk(symbol, start=0, limit=100):
    """
    Reads a chunk of the raw processed CSV dataset for infinite scrolling.
    Returns a dictionary suitable for JSON serialization.
    """
    pattern = os.path.join('data', f"{symbol}_*_to_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        return {"error": f"No dataset found for {symbol}", "data": [], "total": 0}
        
    latest_file = sorted(files)[-1]
    
    try:
        # Read the file
        df = pd.read_csv(latest_file)
        total_rows = len(df)
        
        # Apply pagination slice
        start_idx = int(start)
        limit_idx = int(limit)
        chunk = df.iloc[start_idx : start_idx + limit_idx]
        
        # Format dates nicely if Stock_Timestamp exists
        if 'Stock_Timestamp' in chunk.columns:
            # We assume it's already a string or can be cast
            pass
            
        # Convert to list of dicts and handle NaNs securely
        chunk = chunk.fillna("N/A")
        records = chunk.to_dict('records')
        
        # Determine the columns to display (prioritize important ones)
        all_cols = list(df.columns)
        
        return {
            "symbol": symbol,
            "total": total_rows,
            "start": start_idx,
            "limit": limit_idx,
            "columns": all_cols,
            "data": records
        }
    except Exception as e:
        logger.error(f"Failed to read dataset chunk for {symbol}: {e}")
        return {"error": str(e), "data": [], "total": 0}

def get_predictions(symbol, model_type, mode):
    """
    Returns a JSON object for the frontend to render.
    It returns the paths to the .png visualizations and the textual report.
    """
    pattern = os.path.join('visualizations', f"{symbol}_{model_type}_{mode}_*_v*_*")
    files = glob.glob(pattern)
    
    if not files:
        return json.dumps({"error": f"No visualizations found for {symbol} {model_type} {mode}."})
        
    # Extract the bases (e.g. 'visualizations/MSFT_TFT_Normal_20260219_v1')
    bases = set()
    for f in files:
        # Match everything up to _v<number>
        match = re.search(r'(.*_v\d+)_', f.replace('\\', '/'))
        if match:
            bases.add(match.group(1))
            
    if not bases:
        return json.dumps({"error": "No valid versioned visualizations found."})
        
    # Sort to get the latest base
    latest_base = sorted(list(bases))[-1]
    base_name = os.path.basename(latest_base)
    
    equity_img = f"{base_name}_equity.png"
    cm_img = f"{base_name}_cm.png"
    report_txt = f"{latest_base}_report.txt"  # keep full path for file reading
    csv_file = f"{latest_base}_actual_vs_pred.csv"
    
    report_content = ""
    if os.path.exists(report_txt):
        with open(report_txt, 'r', encoding='utf-8') as f:
            report_content = f.read()
            
    # We replace backslashes with forward slashes for web URL usage
    return json.dumps({
        "type": "images",
        "equity": equity_img,
        "cm": cm_img,
        "report": report_content,
        "has_csv": os.path.exists(csv_file)
    })

if __name__ == "__main__":
    # Test snippet
    logging.basicConfig(level=logging.INFO)
    logger.info("Engine Test:")
    pred_json = get_predictions("MSFT", "TFT", "Normal")
    logger.info(f"Plotly JSON Output snippet: {pred_json[:100]}...")
