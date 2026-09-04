import time
from io import StringIO
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf

from logging_utils import get_logger

log = get_logger("data_provider")

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def safe_float(x):
    try:
        return float(x) if x is not None else np.nan
    except Exception:
        return np.nan

def retry(fn, attempts=3, base_sleep=1.0):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(base_sleep * (i + 1))
    raise last

def get_sp500_universe():
    try:
        def op():
            req = Request(
                WIKI_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 AppleWebKit/605.1.15 Safari/604.1",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
            with urlopen(req, timeout=20) as response:
                html = response.read().decode("utf-8", errors="ignore")

            tables = pd.read_html(StringIO(html))
            if not tables:
                raise RuntimeError("No tables found in S&P 500 source")

            t = tables[0]
            required = {"Symbol", "Security", "GICS Sector"}
            if not required.issubset(set(t.columns)):
                raise RuntimeError(f"Unexpected S&P 500 table columns: {list(t.columns)}")

            t = t[["Symbol", "Security", "GICS Sector"]].copy()
            t["Ticker"] = t["Symbol"].astype(str).str.replace(".", "-", regex=False)
            return t.dropna(subset=["Ticker"]).drop_duplicates("Ticker")

        return retry(op, attempts=3, base_sleep=1.5)
    except Exception as e:
        log.exception("Universe load failed: %s", e)
        return pd.DataFrame(columns=["Symbol", "Security", "GICS Sector", "Ticker"])

def download_prices(tickers, period="2y"):
    if not tickers:
        return pd.DataFrame()
    try:
        return retry(lambda: yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True
        ), attempts=2)
    except Exception as e:
        log.exception("Price download failed: %s", e)
        return pd.DataFrame()

def extract_ohlcv(batch, ticker):
    try:
        if batch.empty:
            return None
        if isinstance(batch.columns, pd.MultiIndex):
            d = pd.DataFrame({k: batch[k][ticker] for k in ["Open", "High", "Low", "Close", "Volume"]})
        else:
            d = batch[["Open", "High", "Low", "Close", "Volume"]].copy()
        d = d.dropna()
        if len(d) < 30 or (d["Close"] <= 0).any():
            return None
        return d
    except Exception:
        return None

def get_ticker_bundle(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info if isinstance(t.info, dict) else {}
        return t, info
    except Exception as e:
        log.warning("Ticker bundle failed %s: %s", ticker, e)
        return None, {}

def latest_price(ticker):
    try:
        h = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if h.empty:
            return np.nan
        c = h["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        return float(c.dropna().iloc[-1])
    except Exception:
        return np.nan
