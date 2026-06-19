import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings

warnings.filterwarnings("ignore")

def create_and_save_visualizations(ticker, input_path, output_dir):
    try:
        merged = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"[{ticker}] Data file not found at {input_path}. Skipping.")
        return

    merged['Stock_Timestamp'] = pd.to_datetime(merged['Stock_Timestamp'], utc=True)

    # By default, take the last 1 day of available data in the frame for a clean visualization
    end_date = merged['Stock_Timestamp'].max()
    start_date = end_date - pd.Timedelta(days=1)

    merged_filtered = merged[(merged['Stock_Timestamp'] >= start_date) &
                             (merged['Stock_Timestamp'] <= end_date)].copy()

    if merged_filtered.empty:
        print(f"[{ticker}] Not enough data in the last 1 day for visualization. Skipping.")
        return

    # Create continuous index to perfectly hide overnight/weekend gaps
    merged_filtered = merged_filtered.reset_index(drop=True)
    merged_filtered['idx'] = merged_filtered.index  
    merged_filtered['hover_time'] = merged_filtered['Stock_Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.4, 0.15, 0.15, 0.15, 0.15]
    )

    # ==========================================
    # ROW 1: PRICE CANDLESTICKS & TREND OVERLAYS
    # ==========================================
    fig.add_trace(go.Candlestick(
        x=merged_filtered['idx'],
        open=merged_filtered['Open'],
        high=merged_filtered['High'],
        low=merged_filtered['Low'],
        close=merged_filtered['Close'],
        name='Price (OHLC)',
        text=merged_filtered['hover_time'],
        hovertext=[f"Time: {t}<br>O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f}" 
                   for t, o, h, l, c in zip(merged_filtered['hover_time'], 
                                            merged_filtered['Open'],
                                            merged_filtered['High'],
                                            merged_filtered['Low'],
                                            merged_filtered['Close'])],
        hoverinfo='text'
    ), row=1, col=1)

    # Define all possible trend overlays and their visual styles
    overlays = [
        ('EMA_10', 'rgba(255, 255, 0, 0.8)', 1, 'solid'),       # Yellow
        ('EMA_50', 'rgba(255, 165, 0, 0.8)', 1.5, 'solid'),     # Orange
        ('SMA_10', 'rgba(173, 216, 230, 0.8)', 1, 'solid'),     # Light Blue
        ('SMA_50', 'rgba(0, 0, 255, 0.8)', 1.5, 'solid'),       # Blue
        ('kalman_close', 'rgba(255, 0, 255, 1)', 2, 'solid'),   # Magenta (Thick) - Updated to match your generation script
        ('rolling_high_20', 'rgba(0, 255, 0, 0.5)', 1, 'dash'), # Green Dashed
        ('rolling_low_20', 'rgba(255, 0, 0, 0.5)', 1, 'dash')   # Red Dashed
    ]

    for col_name, color, width, dash_style in overlays:
        if col_name in merged_filtered.columns:
            fig.add_trace(go.Scatter(
                x=merged_filtered['idx'],
                y=merged_filtered[col_name],
                mode='lines',
                line=dict(color=color, width=width, dash=dash_style),
                name=col_name,
                hovertemplate=f"Time: %{{customdata}}<br>{col_name}: %{{y:.2f}}<extra></extra>",
                customdata=merged_filtered['hover_time']
            ), row=1, col=1)


    # ==========================================
    # ROW 2: VOLUME
    # ==========================================
    fig.add_trace(go.Bar(
        x=merged_filtered['idx'],
        y=merged_filtered['Volume'],
        name='Volume',
        marker_color='gray',
        hovertemplate="Time: %{customdata}<br>Volume: %{y:,.0f}<extra></extra>",
        customdata=merged_filtered['hover_time']
    ), row=2, col=1)


    # ==========================================
    # ROW 3: RSI 14
    # ==========================================
    if 'RSI_14' in merged_filtered.columns:
        fig.add_trace(go.Scatter(
            x=merged_filtered['idx'],
            y=merged_filtered['RSI_14'],
            mode='lines',
            line=dict(color='purple', width=1.5),
            name='RSI_14',
            hovertemplate="Time: %{customdata}<br>RSI: %{y:.2f}<extra></extra>",
            customdata=merged_filtered['hover_time']
        ), row=3, col=1)
        fig.add_hline(y=70, line=dict(color='red', width=1, dash='dash'), row=3, col=1)
        fig.add_hline(y=30, line=dict(color='green', width=1, dash='dash'), row=3, col=1)


    # ==========================================
    # ROW 4: MACD
    # ==========================================
    if 'MACD_12_26_9' in merged_filtered.columns:
        fig.add_trace(go.Scatter(
            x=merged_filtered['idx'],
            y=merged_filtered['MACD_12_26_9'],
            mode='lines',
            line=dict(color='blue', width=1.5),
            name='MACD Line',
            hovertemplate="Time: %{customdata}<br>MACD: %{y:.4f}<extra></extra>",
            customdata=merged_filtered['hover_time']
        ), row=4, col=1)

    if 'MACDs_12_26_9' in merged_filtered.columns:
        fig.add_trace(go.Scatter(
            x=merged_filtered['idx'],
            y=merged_filtered['MACDs_12_26_9'],
            mode='lines',
            line=dict(color='orange', width=1),
            name='Signal Line',
            hovertemplate="Time: %{customdata}<br>Signal: %{y:.4f}<extra></extra>",
            customdata=merged_filtered['hover_time']
        ), row=4, col=1)

    if 'MACDh_12_26_9' in merged_filtered.columns:
        # Differentiate histogram colors based on positive/negative
        colors = ['rgba(0, 255, 0, 0.5)' if val >= 0 else 'rgba(255, 0, 0, 0.5)' for val in merged_filtered['MACDh_12_26_9']]
        fig.add_trace(go.Bar(
            x=merged_filtered['idx'],
            y=merged_filtered['MACDh_12_26_9'],
            name='MACD Histogram',
            marker_color=colors,
            hovertemplate="Time: %{customdata}<br>Histogram: %{y:.4f}<extra></extra>",
            customdata=merged_filtered['hover_time']
        ), row=4, col=1)


    # ==========================================
    # ROW 5: KALMAN RESIDUAL (Scale-Invariant)
    # ==========================================
    residual_col = 'kalman_residual' if 'kalman_residual' in merged_filtered.columns else 'kalman_diff'
    
    if residual_col in merged_filtered.columns:
        fig.add_trace(go.Scatter(
            x=merged_filtered['idx'],
            y=merged_filtered[residual_col],
            mode='lines',
            line=dict(color='cyan', width=1.5),
            name='Kalman Residual',
            hovertemplate="Time: %{customdata}<br>Residual: %{y:.6f}<extra></extra>",
            customdata=merged_filtered['hover_time']
        ), row=5, col=1)
        # Add the zero-line anchor
        fig.add_hline(y=0, line=dict(color='white', width=1, dash='dash'), row=5, col=1)

    # ==========================================
    # X-AXIS TIME LABEL FORMATTING
    # ==========================================
    # Distribute ~8 tick marks evenly across the x-axis to show actual times instead of index numbers
    tick_indices = list(range(0, len(merged_filtered), max(1, len(merged_filtered) // 8)))
    tick_texts = [merged_filtered['hover_time'].iloc[i].split(" ")[1] for i in tick_indices] # Just show the HH:MM:SS

    fig.update_layout(
        title=f'[{ticker}] High-Frequency Technical Dashboard: {start_date.date()} → {end_date.date()}',
        height=1300,
        margin=dict(t=120),  # Added top margin to prevent overlapping
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        template='plotly_dark',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1), # Moved legend up
        # Map the bottom-most x-axis to show actual timestamps
        xaxis5=dict(tickmode='array', tickvals=tick_indices, ticktext=tick_texts)
    )
    
    # Fix Y-axis scaling for RSI
    if 'RSI_14' in merged_filtered.columns:
        fig.update_yaxes(range=[0, 100], row=3, col=1)

    output_path = os.path.join(output_dir, f"{ticker}_technical_dashboard.html")
    fig.write_html(output_path)
    print(f"[{ticker}] Dashboard saved successfully to {output_path}")

def generate_for_ticker(ticker):
    """
    Finds the latest data file for the given ticker and generates the technical dashboard.
    """
    import glob
    data_dir = "data"
    output_dir = os.path.join("visualizations", "Technical Indicators")
    os.makedirs(output_dir, exist_ok=True)
    
    pattern = os.path.join(data_dir, f"{ticker}_*_to_*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"[{ticker}] No data file found matching {pattern}. Skipping.")
        return False
        
    input_path = sorted(files)[-1]
    create_and_save_visualizations(ticker, input_path, output_dir)
    return True

def main():
    import sys
    
    # If a specific ticker is passed via command line, just run that one
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        print(f"=== Generating Visualization for {ticker} ===")
        generate_for_ticker(ticker)
        print("=== Visualization Generation Complete ===")
        return

    # Otherwise, run all
    tickers = [
        'MSFT', 'META', 'V', 'JNJ', 
        'WMT', 'JPM', 'MA', 'PG', 'UNH', 
        'HD', 'LLY', 'XOM', 'COST'
    ]

    print(f"=== Generating Multi-Feature Visualizations for {len(tickers)} Companies ===")
    
    for ticker in tickers:
        generate_for_ticker(ticker)
        
    print("=== Visualization Generation Complete ===")

if __name__ == "__main__":
    main()