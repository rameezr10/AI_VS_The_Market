import os
import re
import glob
import threading
import logging
from flask import Flask, jsonify, request

# Import our orchestration utility
import engine

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__, static_folder='static', template_folder='templates')

# Global Dictionary for Async Task Status tracking
# Structure: { "MSFT": { "status": "idle" | "collecting" | "training" | "error", "message": "..." } }
active_tasks = {}

# =============================================================================
# BACKGROUND THREAD HANDLERS
# =============================================================================
def bg_collect(symbol):
    try:
        output_file = engine.trigger_collection(symbol)
        if output_file:
            active_tasks[symbol] = {
                "status": "idle", 
                "message": f"Successfully collected data: {os.path.basename(output_file)}"
            }
        else:
            active_tasks[symbol] = {"status": "error", "message": "Data collection failed or returned None."}
    except Exception as e:
        logger.error(f"Error in bg_collect for {symbol}: {e}")
        active_tasks[symbol] = {"status": "error", "message": str(e)}

def bg_train(symbol, model_type, mode):
    try:
        success = engine.trigger_training(symbol, model_type, mode)
        if success:
            active_tasks[symbol] = {
                "status": "idle", 
                "message": f"Successfully trained {model_type} ({mode})."
            }
        else:
            active_tasks[symbol] = {"status": "error", "message": f"Training failed for {model_type} ({mode})."}
    except Exception as e:
        logger.error(f"Error in bg_train for {symbol}: {e}")
        active_tasks[symbol] = {"status": "error", "message": str(e)}

# =============================================================================
# ASYNCHRONOUS TASK API ROUTES
# =============================================================================
@app.route('/api/collect/<symbol>', methods=['POST'])
def api_collect(symbol):
    current = active_tasks.get(symbol, {}).get("status", "idle")
    if current in ["collecting", "training"]:
        return jsonify({"error": f"{symbol} is currently busy with: {current}"}), 409
    
    active_tasks[symbol] = {"status": "collecting", "message": "Started data collection..."}
    
    thread = threading.Thread(target=bg_collect, args=(symbol,))
    thread.daemon = True # Daemonize to not block server shutdown
    thread.start()
    
    return jsonify({"message": f"Data collection started for {symbol}"}), 202

@app.route('/api/train/<symbol>', methods=['POST'])
def api_train(symbol):
    current = active_tasks.get(symbol, {}).get("status", "idle")
    if current in ["collecting", "training"]:
        return jsonify({"error": f"{symbol} is currently busy with: {current}"}), 409
        
    data = request.json or {}
    model_type = data.get("model_type", "TFT")
    mode = data.get("mode", "Normal")
    
    active_tasks[symbol] = {"status": "training", "message": f"Started training {model_type} in {mode} mode..."}
    
    thread = threading.Thread(target=bg_train, args=(symbol, model_type, mode))
    thread.daemon = True
    thread.start()
    
    return jsonify({"message": f"Training started for {symbol}"}), 202

@app.route('/api/status/<symbol>', methods=['GET'])
def api_status(symbol):
    state = active_tasks.get(symbol, {"status": "idle", "message": "No active tasks."})
    return jsonify(state)

# =============================================================================
# METADATA & DATA INVENTORY API ROUTES
# =============================================================================
@app.route('/')
def render_index():
    from flask import render_template
    return render_template('index.html')

@app.route('/technical')
def render_technical():
    from flask import render_template
    tickers = ["MSFT", "COST", "HD", "JNJ", "JPM", "LLY", "MA", "META", "PG", "UNH", "V", "WMT", "XOM"]
    tickers = sorted(tickers)
    return render_template('technical.html', tickers=tickers)

@app.route('/dashboard')
def render_dashboard():
    from flask import render_template
    # Pass our list of S&P 500 tickers that we processed
    tickers = ["MSFT", "COST", "HD", "JNJ", "JPM", "LLY", "MA", "META", "PG", "UNH", "V", "WMT", "XOM"]
    tickers = sorted(tickers)
    return render_template('dashboard.html', tickers=tickers)

@app.route('/collection')
def render_collection():
    from flask import render_template
    tickers = ["MSFT", "COST", "HD", "JNJ", "JPM", "LLY", "MA", "META", "PG", "UNH", "V", "WMT", "XOM"]
    tickers = sorted(tickers)
    return render_template('collection.html', tickers=tickers)

@app.route('/training')
def render_training():
    from flask import render_template
    tickers = ["MSFT", "COST", "HD", "JNJ", "JPM", "LLY", "MA", "META", "PG", "UNH", "V", "WMT", "XOM"]
    tickers = sorted(tickers)
    return render_template('training.html', tickers=tickers)

@app.route('/dataset')
def render_dataset():
    from flask import render_template
    tickers = ["MSFT", "COST", "HD", "JNJ", "JPM", "LLY", "MA", "META", "PG", "UNH", "V", "WMT", "XOM"]
    tickers = sorted(tickers)
    return render_template('dataset.html', tickers=tickers)

@app.route('/api/dataset/<symbol>')
def api_dataset(symbol):
    from flask import request
    start = request.args.get('start', 0, type=int)
    limit = request.args.get('limit', 100, type=int)
    
    chunk_data = engine.get_dataset_chunk(symbol, start=start, limit=limit)
    return jsonify(chunk_data)

@app.route('/api/inventory/<symbol>', methods=['GET'])
def api_inventory(symbol):
    # Scan /data/ for datasets
    data_files = glob.glob(os.path.join("data", f"{symbol}_*_to_*.csv"))
    datasets = []
    for df in data_files:
        basename = os.path.basename(df)
        match = re.search(r'_(\d{8})_to_(\d{8})\.csv$', basename)
        if match:
            datasets.append({
                "file": basename,
                "start_date": match.group(1),
                "end_date": match.group(2)
            })
            
    # Scan /saved_models/ for trained models
    model_files = glob.glob(os.path.join("saved_models", f"{symbol}_*_*_*_v*.*"))
    models = []
    for mf in model_files:
        basename = os.path.basename(mf)
        # Regex to parse the standardized output string: {SYMBOL}_{MODEL}_{MODE}_{DATE}_v{VERSION}.ext
        match = re.search(rf'^{symbol}_(.*?)_(Normal|Baseline)_(\d{{8}})_v(\d+)\.(pth|ckpt)$', basename)
        if match:
            models.append({
                "file": basename,
                "model_type": match.group(1),
                "mode": match.group(2),
                "date": match.group(3),
                "version": int(match.group(4))
            })
            
    # Sort for convenience (latest first)
    datasets = sorted(datasets, key=lambda x: x["end_date"], reverse=True)
    models = sorted(models, key=lambda x: (x["date"], x["version"]), reverse=True)
            
    return jsonify({
        "symbol": symbol,
        "datasets": datasets, 
        "models": models
    })

@app.route('/api/visualize/<symbol>/<model_type>/<mode>', methods=['GET'])
def api_visualize(symbol, model_type, mode):
    try:
        plot_json_str = engine.get_predictions(symbol, model_type, mode)
        return plot_json_str, 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"Error rendering visualization for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/visualizations/<path:filename>')
def serve_visualizations(filename):
    from flask import send_from_directory
    return send_from_directory('visualizations', filename)

# =============================================================================
# SERVER ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    # Ensure necessary backend directories exist before starting the API
    os.makedirs('data', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Run the Flask App
    app.run(debug=True, host='0.0.0.0', port=5000)
