import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import requests
import pandas as pd
import pandas_ta as ta
import time
import sys
import os
import warnings
from datetime import datetime, timedelta, timezone
import pytz

# Added missing import for KalmanFilter
from pykalman import KalmanFilter

warnings.filterwarnings("ignore")

API_KEY = "IRJ9pvan0exvtjl9UCh9DJDLrLOOwYVo"

def fetch_stock_data(ticker, start_date, end_date):
    multiplier = 1
    timespan = "minute"
    MAX_BARS = 50000
    days_per_chunk = 128

    us_eastern_tz = pytz.timezone("America/New_York")
    market_open = datetime(2023, 1, 1, 9, 30).time()
    market_close = datetime(2023, 1, 1, 16, 0).time()

    logger.info(f"[{ticker}] Fetching 1-minute data from {start_date} to {end_date}")

    df_list = []
    chunk_start = start_date

    with requests.Session() as session:
        while chunk_start <= end_date:
            chunk_end = min(chunk_start + timedelta(days=days_per_chunk - 1), end_date)

            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/"
                f"{multiplier}/{timespan}/{chunk_start}/{chunk_end}"
            )
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": MAX_BARS,
                "apiKey": API_KEY
            }

            logger.info(f"  Fetching: {chunk_start} → {chunk_end}")

            try:
                resp = session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                if not data.get("results"):
                    chunk_start = chunk_end + timedelta(days=1)
                    continue

                df = pd.DataFrame(data["results"])

                df["Datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
                df["Datetime"] = df["Datetime"].dt.tz_convert(us_eastern_tz)

                df.rename(columns={
                    "o": "Open", "h": "High", "l": "Low", "c": "Close",
                    "v": "Volume", "n": "Transactions"
                }, inplace=True)

                df = df[
                    (df["Datetime"].dt.time >= market_open) &
                    (df["Datetime"].dt.time <= market_close)
                ].copy()

                df = df[["Datetime", "Open", "High", "Low", "Close", "Volume", "Transactions"]]
                df_list.append(df)

            except requests.exceptions.HTTPError as e:
                logger.info(f"  ❌ HTTP ERROR: {e}")
                if e.response.status_code == 429:
                    logger.info("     Rate limit hit → sleeping 60s")
                    time.sleep(60)
                    continue
                if e.response.status_code == 401:
                    logger.info("     INVALID API KEY!")
                    break

            except requests.exceptions.RequestException as e:
                logger.info(f"  ❌ Request failed: {e}")
                logger.info("     Retrying in 10s...")
                time.sleep(10)
                continue

            chunk_start = chunk_end + timedelta(days=1)
            # Sleep 13 seconds natively to avoid rate limiting across 20 companies
            time.sleep(13)

    if not df_list:
        return None

    df_stocks = pd.concat(df_list, ignore_index=True)
    df_stocks["Datetime"] = pd.to_datetime(df_stocks["Datetime"])
    df_stocks = df_stocks.set_index("Datetime").sort_index()
    df_stocks = df_stocks[~df_stocks.index.duplicated(keep="first")]
    
    return df_stocks


def fetch_news_data(ticker, start_date, end_date):
    us_eastern_tz = pytz.timezone("America/New_York")
    base_url = "https://api.polygon.io/v2/reference/news"

    params = {
        "ticker": ticker,
        "order": "asc",
        "published_utc.gte": start_date.isoformat(),
        "published_utc.lte": end_date.isoformat(),
        "limit": 100,
        "apiKey": API_KEY
    }

    all_news = []
    next_url = None
    logger.info(f"[{ticker}] Fetching news from {start_date} to {end_date}")

    with requests.Session() as session:
        while True:
            attempt = 0
            max_attempts = 5

            while attempt < max_attempts:
                try:
                    if next_url:
                        url_to_call = next_url
                        if "apiKey=" not in url_to_call:
                            sep = "&" if "?" in url_to_call else "?"
                            url_to_call = f"{url_to_call}{sep}apiKey={API_KEY}"
                        resp = session.get(url_to_call, timeout=30)
                    else:
                        resp = session.get(base_url, params=params, timeout=30)

                    resp.raise_for_status()
                    data = resp.json()
                    break

                except requests.exceptions.HTTPError as e:
                    status = e.response.status_code
                    if status == 401:
                        logger.info(f"  ❌ FATAL: API Key invalid.")
                        return None
                    if status == 429:
                        logger.info("  ❌ Rate limit hit. Sleeping 60s...")
                        time.sleep(60)
                        attempt += 1
                        continue
                    break

                except requests.exceptions.RequestException as e:
                    logger.info(f"  ❌ Request Error: {e}. Retrying...")
                    time.sleep(10)
                    attempt += 1
                    continue

            if attempt == max_attempts:
                logger.info(f"  ❌ Failed after {max_attempts} attempts.")
                break

            results = data.get("results", [])
            if not results:
                break

            for article in results:
                all_news.append({
                    "Published_UTC": article.get("published_utc"),
                    "Title": article.get("title"),
                    "Description": article.get("description"),
                    "Sentiment": (
                        article.get("insights", [{}])[0].get("sentiment")
                        if article.get("insights") else None
                    )
                })

            next_token_candidate = data.get("next")
            next_url_candidate = data.get("next_url")

            if next_url_candidate:
                next_url = next_url_candidate
                params = None
            elif next_token_candidate:
                params = {
                    "ticker": ticker,
                    "order": "asc",
                    "published_utc.gte": start_date.isoformat(),
                    "published_utc.lte": end_date.isoformat(),
                    "limit": 100,
                    "cursor": next_token_candidate,
                    "apiKey": API_KEY
                }
                next_url = None
            else:
                break

            # Sleep 13 seconds natively to respect Polygon's 5 calls/min limit (60s/5 = 12s)
            time.sleep(13)

    df_news = pd.DataFrame(all_news)
    if not df_news.empty:
        df_news["Published_UTC"] = pd.to_datetime(df_news["Published_UTC"], utc=True)
        df_news["Datetime"] = df_news["Published_UTC"].dt.tz_convert(us_eastern_tz)
        df_news = df_news.sort_values("Datetime").reset_index(drop=True)
        return df_news
    
    return None


def process_and_merge_data(df_stocks, df_news):
    df_stocks_sorted = df_stocks.sort_index().reset_index()
    
    if df_news is not None and not df_news.empty:
        df_news_sorted = df_news.sort_values("Datetime").reset_index(drop=True)
        df_news_sorted["News_Time"] = df_news_sorted["Datetime"]

        merged_df = pd.merge_asof(
            df_stocks_sorted,
            df_news_sorted,
            on="Datetime",
            direction="backward"
        )
        merged_df["News_Age_Minutes"] = (
            (merged_df["Datetime"] - merged_df["News_Time"]).dt.total_seconds() / 60
        )
    else:
        merged_df = df_stocks_sorted.copy()
        for col in ["Title", "Description", "Sentiment", "News_Time", "News_Age_Minutes"]:
            merged_df[col] = None

    merged_df.rename(columns={"Datetime": "Stock_Timestamp"}, inplace=True)
    return merged_df


def apply_features(merged_df):
    merged = merged_df.copy()
    
    # 1. Standard Indicators
    merged.ta.sma(length=10, append=True) 
    merged.ta.sma(length=50, append=True) 
    merged.ta.ema(length=10, append=True) 
    merged.ta.ema(length=50, append=True) 
    merged.ta.macd(append=True)
    merged.ta.rsi(length=14, append=True) 

    # 2. Naive Filter features
    k = 20 
    merged[f'rolling_high_{k}'] = merged['High'].rolling(window=k).max()
    merged[f'rolling_low_{k}'] = merged['Low'].rolling(window=k).min()
    merged['return_1m'] = merged['Close'].pct_change() * 100

    logger.info("Applying Kalman Filter...")
    close_prices = merged['Close'].values
    try:
        
        kf = KalmanFilter(
            initial_state_mean=close_prices[0],
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=0.01,
            transition_matrices=[1]
        )
        
        kf = kf.em(close_prices, n_iter=5)
        (smoothed_state_means, _) = kf.smooth(close_prices)
        
        merged['kalman_close'] = smoothed_state_means
        merged['kalman_diff'] = merged['Close'] - merged['kalman_close']
        
    except Exception as e:
        logger.info(f"  > Kalman Filter failed: {e}. Skipping this feature.")

    merged.dropna(inplace=True)
    merged = merged.reset_index(drop=True)
    return merged


def run_collection(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    logger.info(f"=== Starting Data Collection Pipeline for {ticker} ===")
    logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")
    
    try:
        df_stocks = fetch_stock_data(ticker, start_date.date(), end_date.date())
        if df_stocks is None or df_stocks.empty:
            logger.info(f"[{ticker}] No stock data fetched. Skipping.")
            return None

        df_news = fetch_news_data(ticker, start_date.date(), end_date.date())
        
        merged_df = process_and_merge_data(df_stocks, df_news)
        
        final_df = apply_features(merged_df)
        
        os.makedirs("data", exist_ok=True)
        date_str_start = start_date.strftime('%Y%m%d')
        date_str_end = end_date.strftime('%Y%m%d')
        output_file = os.path.join("data", f"{ticker}_{date_str_start}_to_{date_str_end}.csv")
        final_df.to_csv(output_file, index=False)
        logger.info(f"✅ [{ticker}] Successfully saved to {output_file} | Rows: {len(final_df)}")
        return output_file
        
    except Exception as e:
        logger.error(f"❌ [{ticker}] Pipeline encountered an unexpected error: {e}")
        return None

if __name__ == "__main__":
    run_collection("MSFT")