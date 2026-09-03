import numpy as np
import pandas as pd
import yfinance as yf

def regime_snapshot():
    try:
        d=yf.download(["SPY","QQQ","^VIX"],period="2y",auto_adjust=True,progress=False)
        spy=d["Close"]["SPY"].dropna()
        qqq=d["Close"]["QQQ"].dropna()
        vix=d["Close"]["^VIX"].dropna()
        spy_r=spy.pct_change()
        qqq_r=qqq.pct_change()
        return {
            "spy_20d":float((spy.iloc[-1]/spy.iloc[-21]-1)*100),
            "qqq_20d":float((qqq.iloc[-1]/qqq.iloc[-21]-1)*100),
            "vix":float(vix.iloc[-1]),
            "spy_vol_20d":float(spy_r.tail(20).std()*np.sqrt(252)*100),
            "qqq_vol_20d":float(qqq_r.tail(20).std()*np.sqrt(252)*100),
        }
    except Exception:
        return {}

def drift_score(snap):
    if not snap: return 50.0
    score=0
    score+=25 if snap.get("spy_20d",0)>0 else 0
    score+=25 if snap.get("qqq_20d",0)>0 else 0
    score+=25 if snap.get("vix",99)<25 else (12 if snap.get("vix",99)<32 else 0)
    avgvol=(snap.get("spy_vol_20d",50)+snap.get("qqq_vol_20d",50))/2
    score+=25 if avgvol<25 else (12 if avgvol<35 else 0)
    return float(score)
