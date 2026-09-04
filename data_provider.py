import time
from io import StringIO
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf

from logging_utils import get_logger

log = get_logger("data_provider")

RAW_SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

# Emergency fallback so the scanner never collapses to 0 rows just because
# the universe source is temporarily unavailable.
FALLBACK = [
    ("AAPL","Apple","Information Technology"),("MSFT","Microsoft","Information Technology"),
    ("NVDA","NVIDIA","Information Technology"),("AMZN","Amazon","Consumer Discretionary"),
    ("GOOGL","Alphabet Class A","Communication Services"),("GOOG","Alphabet Class C","Communication Services"),
    ("META","Meta Platforms","Communication Services"),("BRK-B","Berkshire Hathaway","Financials"),
    ("LLY","Eli Lilly","Health Care"),("AVGO","Broadcom","Information Technology"),
    ("TSLA","Tesla","Consumer Discretionary"),("JPM","JPMorgan Chase","Financials"),
    ("V","Visa","Financials"),("WMT","Walmart","Consumer Staples"),
    ("MA","Mastercard","Financials"),("XOM","Exxon Mobil","Energy"),
    ("UNH","UnitedHealth Group","Health Care"),("COST","Costco","Consumer Staples"),
    ("ORCL","Oracle","Information Technology"),("NFLX","Netflix","Communication Services"),
    ("HD","Home Depot","Consumer Discretionary"),("PG","Procter & Gamble","Consumer Staples"),
    ("JNJ","Johnson & Johnson","Health Care"),("ABBV","AbbVie","Health Care"),
    ("BAC","Bank of America","Financials"),("KO","Coca-Cola","Consumer Staples"),
    ("CRM","Salesforce","Information Technology"),("CVX","Chevron","Energy"),
    ("MRK","Merck","Health Care"),("AMD","Advanced Micro Devices","Information Technology"),
    ("PEP","PepsiCo","Consumer Staples"),("CSCO","Cisco","Information Technology"),
    ("ACN","Accenture","Information Technology"),("TMO","Thermo Fisher","Health Care"),
    ("LIN","Linde","Materials"),("MCD","McDonald's","Consumer Discretionary"),
    ("ABT","Abbott Laboratories","Health Care"),("DIS","Walt Disney","Communication Services"),
    ("WFC","Wells Fargo","Financials"),("IBM","IBM","Information Technology"),
    ("INTU","Intuit","Information Technology"),("QCOM","Qualcomm","Information Technology"),
    ("CAT","Caterpillar","Industrials"),("GE","GE Aerospace","Industrials"),
    ("AMGN","Amgen","Health Care"),("NOW","ServiceNow","Information Technology"),
    ("TXN","Texas Instruments","Information Technology"),("ISRG","Intuitive Surgical","Health Care"),
    ("BKNG","Booking Holdings","Consumer Discretionary"),("SPGI","S&P Global","Financials"),
    ("AXP","American Express","Financials"),("LOW","Lowe's","Consumer Discretionary"),
    ("HON","Honeywell","Industrials"),("PFE","Pfizer","Health Care"),
    ("AMAT","Applied Materials","Information Technology"),("UPS","UPS","Industrials"),
    ("RTX","RTX","Industrials"),("GS","Goldman Sachs","Financials"),
    ("SCHW","Charles Schwab","Financials"),("BLK","BlackRock","Financials"),
]

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

def _fallback_universe():
    return pd.DataFrame(FALLBACK, columns=["Ticker","Security","GICS Sector"]).assign(Symbol=lambda x: x["Ticker"])[["Symbol","Security","GICS Sector","Ticker"]]

def get_sp500_universe():
    try:
        def op():
            req = Request(
                RAW_SP500_URL,
                headers={
                    "User-Agent":"Mozilla/5.0",
                    "Accept":"text/csv,*/*;q=0.8"
                }
            )
            with urlopen(req, timeout=20) as r:
                csv_text = r.read().decode("utf-8", errors="ignore")

            t = pd.read_csv(StringIO(csv_text))
            required = {"Symbol","Security","GICS Sector"}
            if not required.issubset(t.columns):
                raise RuntimeError(f"Unexpected columns: {list(t.columns)}")
            t = t[["Symbol","Security","GICS Sector"]].copy()
            t["Ticker"] = t["Symbol"].astype(str).str.replace(".", "-", regex=False)
            t = t.dropna(subset=["Ticker"]).drop_duplicates("Ticker")
            if len(t) < 400:
                raise RuntimeError(f"Universe unexpectedly small: {len(t)}")
            return t

        return retry(op, attempts=3, base_sleep=1.5)

    except Exception as e:
        log.warning("Primary S&P universe source failed; using fallback: %s", e)
        return _fallback_universe()

def download_prices(tickers, period="2y"):
    if not tickers:
        return pd.DataFrame()
    try:
        return retry(
            lambda: yf.download(
                tickers=tickers,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=True,
            ),
            attempts=2,
        )
    except Exception as e:
        log.exception("Price download failed: %s", e)
        return pd.DataFrame()

def extract_ohlcv(batch, ticker):
    try:
        if batch.empty:
            return None
        if isinstance(batch.columns, pd.MultiIndex):
            d = pd.DataFrame({k: batch[k][ticker] for k in ["Open","High","Low","Close","Volume"]})
        else:
            d = batch[["Open","High","Low","Close","Volume"]].copy()
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
            c = c.iloc[:,0]
        return float(c.dropna().iloc[-1])
    except Exception:
        return np.nan
