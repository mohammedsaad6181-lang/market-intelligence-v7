import numpy as np
import pandas as pd

def norm(x,lo,hi,invert=False,neutral=50.0):
    if x is None or pd.isna(x) or hi==lo: return neutral
    v=float(np.clip((float(x)-lo)/(hi-lo),0,1)*100)
    return 100-v if invert else v

def rsi(s,n=14):
    d=s.diff()
    up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=up/dn.replace(0,np.nan)
    return 100-100/(1+rs)

def atr(df,n=14):
    pc=df["Close"].shift(1)
    tr=pd.concat([(df["High"]-df["Low"]).abs(),(df["High"]-pc).abs(),(df["Low"]-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def adx(df,n=14):
    up=df["High"].diff(); down=-df["Low"].diff()
    plus=up.where((up>down)&(up>0),0.0)
    minus=down.where((down>up)&(down>0),0.0)
    tr=atr(df,n)
    pdi=100*plus.ewm(alpha=1/n,adjust=False).mean()/tr.replace(0,np.nan)
    mdi=100*minus.ewm(alpha=1/n,adjust=False).mean()/tr.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean()

def technical_metrics(df):
    if df is None or len(df)<220: return None
    x=df.copy(); c=x["Close"]
    x["SMA20"]=c.rolling(20).mean(); x["SMA50"]=c.rolling(50).mean(); x["SMA200"]=c.rolling(200).mean()
    x["EMA12"]=c.ewm(span=12,adjust=False).mean(); x["EMA26"]=c.ewm(span=26,adjust=False).mean()
    x["MACD"]=x["EMA12"]-x["EMA26"]; x["MACDsig"]=x["MACD"].ewm(span=9,adjust=False).mean()
    x["RSI"]=rsi(c); x["ATR"]=atr(x); x["ADX"]=adx(x)
    q=x.iloc[-1]

    def ret(n): return (c.iloc[-1]/c.iloc[-n-1]-1)*100 if len(c)>n else np.nan
    r20,r60,r120,r252=ret(20),ret(60),ret(120),ret(min(252,len(c)-2))
    vol=c.pct_change().tail(60).std()*np.sqrt(252)*100
    atrp=q["ATR"]/q["Close"]*100
    v20=x["Volume"].tail(20).mean(); v60=x["Volume"].tail(60).mean()
    vr=v20/v60 if pd.notna(v60) and v60>0 else np.nan
    trend=(22*(q["Close"]>q["SMA20"])+22*(q["SMA20"]>q["SMA50"])+28*(q["SMA50"]>q["SMA200"])+28*(q["Close"]>q["SMA200"]))
    macds=90 if q["MACD"]>q["MACDsig"] and q["MACD"]>0 else (65 if q["MACD"]>q["MACDsig"] else 30)
    rr=q["RSI"]; rs=95 if 50<=rr<=68 else (75 if 42<=rr<50 or 68<rr<=74 else 35)
    technical=.38*trend+.23*macds+.20*rs+.12*norm(q["ADX"],15,40)+.07*norm(vr,.75,1.4)
    momentum=.40*norm(r20,-12,18)+.30*norm(r60,-20,35)+.20*norm(r120,-30,55)+.10*norm(r252,-40,90)
    risk=.45*norm(vol,18,75,invert=True)+.35*norm(atrp,1,6,invert=True)+.20*(100 if q["Close"]>q["SMA200"] else 25)
    return {
        "price":float(q["Close"]),"ret20":r20,"ret60":r60,"ret120":r120,"ret252":r252,
        "rsi":float(rr),"adx":float(q["ADX"]),"atr":float(q["ATR"]),"atr_pct":float(atrp),
        "volatility":float(vol),"volume_ratio":float(vr) if pd.notna(vr) else np.nan,
        "technical":float(technical),"momentum":float(momentum),"risk":float(risk),
        "sma20":float(q["SMA20"]),"sma50":float(q["SMA50"]),"sma200":float(q["SMA200"]),
        "macd":float(q["MACD"]),"macd_signal":float(q["MACDsig"]),"above200":bool(q["Close"]>q["SMA200"])
    }
