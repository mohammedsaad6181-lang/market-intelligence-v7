import time
import numpy as np
import pandas as pd
import yfinance as yf
from logging_utils import get_logger

log=get_logger("data_provider")

def safe_float(x):
    try: return float(x) if x is not None else np.nan
    except Exception: return np.nan

def retry(fn,attempts=3,base_sleep=1.0):
    last=None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last=e
            time.sleep(base_sleep*(i+1))
    raise last

def get_sp500_universe():
    try:
        def op():
            t=pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
            t=t[["Symbol","Security","GICS Sector"]].copy()
            t["Ticker"]=t["Symbol"].str.replace(".","-",regex=False)
            return t
        return retry(op)
    except Exception as e:
        log.exception("Universe load failed: %s",e)
        return pd.DataFrame(columns=["Symbol","Security","GICS Sector","Ticker"])

def download_prices(tickers,period="2y"):
    if not tickers: return pd.DataFrame()
    try:
        return retry(lambda: yf.download(
            tickers=tickers,period=period,interval="1d",
            auto_adjust=True,progress=False,group_by="column",threads=True
        ),attempts=2)
    except Exception as e:
        log.exception("Price download failed: %s",e)
        return pd.DataFrame()

def extract_ohlcv(batch,ticker):
    try:
        if batch.empty: return None
        if isinstance(batch.columns,pd.MultiIndex):
            d=pd.DataFrame({k:batch[k][ticker] for k in ["Open","High","Low","Close","Volume"]})
        else:
            d=batch[["Open","High","Low","Close","Volume"]].copy()
        d=d.dropna()
        if len(d)<30 or (d["Close"]<=0).any(): return None
        return d
    except Exception:
        return None

def get_ticker_bundle(ticker):
    try:
        t=yf.Ticker(ticker)
        info=t.info if isinstance(t.info,dict) else {}
        return t,info
    except Exception as e:
        log.warning("Ticker bundle failed %s: %s",ticker,e)
        return None,{}

def latest_price(ticker):
    try:
        h=yf.download(ticker,period="5d",auto_adjust=True,progress=False)
        if h.empty: return np.nan
        c=h["Close"]
        if isinstance(c,pd.DataFrame): c=c.iloc[:,0]
        return float(c.dropna().iloc[-1])
    except Exception:
        return np.nan
