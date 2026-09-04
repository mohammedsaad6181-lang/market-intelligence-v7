import time
from io import StringIO
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf

from logging_utils import get_logger

log = get_logger("data_provider")

RAW_SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

FALLBACK = [
    ("AAPL","Apple","Information Technology"),("MSFT","Microsoft","Information Technology"),
    ("NVDA","NVIDIA","Information Technology"),("AMZN","Amazon","Consumer Discretionary"),
    ("GOOGL","Alphabet Class A","Communication Services"),("META","Meta Platforms","Communication Services"),
    ("BRK-B","Berkshire Hathaway","Financials"),("LLY","Eli Lilly","Health Care"),
    ("AVGO","Broadcom","Information Technology"),("TSLA","Tesla","Consumer Discretionary"),
    ("JPM","JPMorgan Chase","Financials"),("V","Visa","Financials"),
    ("WMT","Walmart","Consumer Staples"),("MA","Mastercard","Financials"),
    ("XOM","Exxon Mobil","Energy"),("UNH","UnitedHealth Group","Health Care"),
    ("COST","Costco","Consumer Staples"),("ORCL","Oracle","Information Technology"),
    ("NFLX","Netflix","Communication Services"),("HD","Home Depot","Consumer Discretionary"),
    ("PG","Procter & Gamble","Consumer Staples"),("JNJ","Johnson & Johnson","Health Care"),
    ("ABBV","AbbVie","Health Care"),("BAC","Bank of America","Financials"),
    ("KO","Coca-Cola","Consumer Staples"),("CRM","Salesforce","Information Technology"),
    ("CVX","Chevron","Energy"),("MRK","Merck","Health Care"),
    ("AMD","Advanced Micro Devices","Information Technology"),("PEP","PepsiCo","Consumer Staples"),
    ("CSCO","Cisco","Information Technology"),("ACN","Accenture","Information Technology"),
    ("TMO","Thermo Fisher","Health Care"),("LIN","Linde","Materials"),
    ("MCD","McDonald's","Consumer Discretionary"),("ABT","Abbott Laboratories","Health Care"),
    ("DIS","Walt Disney","Communication Services"),("WFC","Wells Fargo","Financials"),
    ("IBM","IBM","Information Technology"),("INTU","Intuit","Information Technology"),
    ("QCOM","Qualcomm","Information Technology"),("CAT","Caterpillar","Industrials"),
    ("GE","GE Aerospace","Industrials"),("AMGN","Amgen","Health Care"),
    ("NOW","ServiceNow","Information Technology"),("TXN","Texas Instruments","Information Technology"),
    ("ISRG","Intuitive Surgical","Health Care"),("BKNG","Booking Holdings","Consumer Discretionary"),
    ("SPGI","S&P Global","Financials"),("AXP","American Express","Financials"),
    ("LOW","Lowe's","Consumer Discretionary"),("HON","Honeywell","Industrials"),
    ("PFE","Pfizer","Health Care"),("AMAT","Applied Materials","Information Technology"),
    ("UPS","UPS","Industrials"),("RTX","RTX","Industrials"),
    ("GS","Goldman Sachs","Financials"),("SCHW","Charles Schwab","Financials"),
    ("BLK","BlackRock","Financials"),("PM","Philip Morris","Consumer Staples"),
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
    t = pd.DataFrame(FALLBACK, columns=["Ticker","Security","GICS Sector"])
    t["Symbol"] = t["Ticker"]
    return t[["Symbol","Security","GICS Sector","Ticker"]]

def get_sp500_universe():
    try:
        req = Request(RAW_SP500_URL, headers={"User-Agent":"Mozilla/5.0","Accept":"text/csv,*/*"})
        with urlopen(req, timeout=20) as r:
            txt = r.read().decode("utf-8", errors="ignore")
        t = pd.read_csv(StringIO(txt))
        required = {"Symbol","Security","GICS Sector"}
        if not required.issubset(t.columns):
            raise RuntimeError(f"Unexpected universe columns: {list(t.columns)}")
        t = t[["Symbol","Security","GICS Sector"]].copy()
        t["Ticker"] = t["Symbol"].astype(str).str.replace(".", "-", regex=False)
        t = t.dropna(subset=["Ticker"]).drop_duplicates("Ticker")
        if len(t) < 400:
            raise RuntimeError(f"Universe unexpectedly small: {len(t)}")
        log.info("Universe loaded: %d tickers", len(t))
        return t
    except Exception as e:
        fb = _fallback_universe()
        log.warning("Primary universe failed (%s). Using fallback %d tickers.", e, len(fb))
        return fb

def _download_chunk(chunk, period):
    return yf.download(
        tickers=chunk,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )

def download_prices(tickers, period="2y", chunk_size=25):
    if not tickers:
        return pd.DataFrame()

    tickers = list(dict.fromkeys([str(t) for t in tickers if t]))
    pieces = []
    successful = 0

    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start:start+chunk_size]
        try:
            part = retry(lambda: _download_chunk(chunk, period), attempts=3, base_sleep=1.5)
            if part is not None and not part.empty:
                pieces.append(part)
                successful += len(chunk)
                log.info("Price chunk OK: %d-%d (%d tickers)", start+1, start+len(chunk), len(chunk))
            else:
                log.warning("Empty price chunk: %s", ",".join(chunk[:5]))
        except Exception as e:
            log.warning("Price chunk failed (%s): %s", ",".join(chunk[:5]), e)
        time.sleep(0.35)

    if not pieces:
        log.error("All bulk price chunks failed. Trying fallback universe individually.")
        fallback_tickers = _fallback_universe()["Ticker"].tolist()
        single_parts = []
        for ticker in fallback_tickers[:30]:
            try:
                h = retry(lambda: yf.download(
                    ticker, period=period, interval="1d", auto_adjust=True,
                    progress=False, threads=False
                ), attempts=2, base_sleep=1.0)
                if h is None or h.empty:
                    continue
                if not isinstance(h.columns, pd.MultiIndex):
                    h.columns = pd.MultiIndex.from_product([h.columns, [ticker]])
                single_parts.append(h)
            except Exception as e:
                log.warning("Single ticker failed %s: %s", ticker, e)
            time.sleep(0.2)
        if single_parts:
            combined = pd.concat(single_parts, axis=1)
            log.info("Individual fallback succeeded for %d tickers", len(single_parts))
            return combined
        return pd.DataFrame()

    # Concatenate chunks by columns. Duplicate columns can happen if a provider
    # unexpectedly repeats a ticker; keep the first occurrence.
    out = pd.concat(pieces, axis=1)
    if isinstance(out.columns, pd.MultiIndex):
        out = out.loc[:, ~out.columns.duplicated()]
    log.info("Price download completed. Requested=%d, chunk_successâ%d", len(tickers), successful)
    return out

def extract_ohlcv(batch, ticker):
    try:
        if batch is None or batch.empty:
            return None

        if isinstance(batch.columns, pd.MultiIndex):
            # Normal yfinance multi-ticker layout: level0=OHLCV, level1=ticker.
            fields = set(batch.columns.get_level_values(0))
            tickers = set(batch.columns.get_level_values(1))
            if ticker in tickers and {"Open","High","Low","Close","Volume"}.issubset(fields):
                d = pd.DataFrame({
                    k: pd.to_numeric(batch[(k, ticker)], errors="coerce")
                    for k in ["Open","High","Low","Close","Volume"]
                })
            else:
                return None
        else:
            needed=["Open","High","Low","Close","Volume"]
            if not set(needed).issubset(batch.columns):
                return None
            d = batch[needed].copy()

        d = d.replace([np.inf,-np.inf], np.nan).dropna()
        if len(d) < 220:
            return None
        if (d["Close"] <= 0).any():
            return None
        return d
    except Exception as e:
        log.debug("extract_ohlcv failed %s: %s", ticker, e)
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
        h = yf.download(ticker, period="5d", auto_adjust=True, progress=False, threads=False)
        if h.empty:
            return np.nan
        c = h["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:,0]
        return float(c.dropna().iloc[-1])
    except Exception:
        return np.nan
