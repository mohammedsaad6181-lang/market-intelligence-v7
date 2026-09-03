import numpy as np
import pandas as pd
import yfinance as yf
from config import SETTINGS
from data_provider import get_sp500_universe,download_prices,extract_ohlcv,get_ticker_bundle
from indicators import technical_metrics,norm
from fundamentals import fundamentals,valuation,analyst_metrics
from altdata import news_signals,earnings_signals,insider_institutional,options_signals
from quality import confidence_score
from regime import regime_snapshot, drift_score
from logging_utils import get_logger

log=get_logger("engine")

def market_regime():
    try:
        d=yf.download(["SPY","QQQ","^VIX"],period="1y",auto_adjust=True,progress=False)
        spy=d["Close"]["SPY"].dropna(); qqq=d["Close"]["QQQ"].dropna(); vix=d["Close"]["^VIX"].dropna()
        score=0
        score+=35 if spy.iloc[-1]>spy.rolling(200).mean().iloc[-1] else 0
        score+=20 if spy.rolling(50).mean().iloc[-1]>spy.rolling(200).mean().iloc[-1] else 0
        score+=25 if qqq.iloc[-1]>qqq.rolling(200).mean().iloc[-1] else 0
        score+=10 if qqq.rolling(50).mean().iloc[-1]>qqq.rolling(200).mean().iloc[-1] else 0
        vx=float(vix.iloc[-1]); score+=10 if vx<25 else (5 if vx<32 else 0)
        return {"score":score,"label":"Risk-On" if score>=75 else ("Neutral" if score>=45 else "Risk-Off"),"vix":vx}
    except Exception as e:
        log.warning("Regime fallback: %s",e)
        return {"score":50,"label":"Unknown","vix":np.nan}

def sector_relative(df):
    df=df.copy(); df["sector_value_score"]=50.0
    for _,sub in df.groupby("Sector"):
        med=sub["forward_pe"].median(skipna=True)
        if pd.isna(med) or med<=0: continue
        for i,r in sub.iterrows():
            f=r["forward_pe"]
            if pd.notna(f) and f>0: df.at[i,"sector_value_score"]=norm(f/med,.65,1.55,invert=True)
    return df

def apply_scores(df,mkt):
    df=df.copy()
    df["LongScore"]=(.18*df["growth"]+.17*df["quality"]+.13*df["valuation"]+.10*df["sector_value_score"]+
                     .10*df["cashflow"]+.08*df["balance"]+.07*df["earnings_score"]+.05*df["analyst"]+
                     .04*df["insider_score"]+.03*df["news_score"]+.03*df["risk"]+.02*df["technical"])
    df["ShortScore"]=(.28*df["technical"]+.24*df["momentum"]+.12*df["risk"]+.08*df["news_score"]+
                      .07*df["earnings_score"]+.06*df["options_score"]+.06*df["analyst"]+.04*df["growth"]+
                      .03*df["valuation"]+.02*mkt)
    df.loc[df["rsi"]>80,"ShortScore"]-=8
    df.loc[df["volatility"]>75,"ShortScore"]-=7
    df.loc[~df["above200"],"ShortScore"]-=10
    df.loc[(df["next_earnings_days"]>=0)&(df["next_earnings_days"]<=5),"ShortScore"]-=8
    df.loc[(df["fcf_yield"].notna())&(df["fcf_yield"]<0),"LongScore"]-=5
    df.loc[(df["debt_to_equity"].notna())&(df["debt_to_equity"]>300),"LongScore"]-=6
    df.loc[(df["earnings_growth"].notna())&(df["earnings_growth"]<-15),"LongScore"]-=8
    df["LongScore"]=df["LongScore"].clip(0,100); df["ShortScore"]=df["ShortScore"].clip(0,100)
    return df

def scan(deep_candidates=None,option_candidates=None):
    deep_candidates=deep_candidates or SETTINGS.deep_candidates
    option_candidates=option_candidates or SETTINGS.option_candidates
    u=get_sp500_universe()
    if u.empty: return pd.DataFrame(),market_regime()
    b=download_prices(u["Ticker"].tolist())
    rows=[]
    for _,m in u.iterrows():
        d=extract_ohlcv(b,m["Ticker"]); t=technical_metrics(d)
        if not t: continue
        adv=(d["Close"].tail(20)*d["Volume"].tail(20)).mean()
        if pd.isna(adv) or adv<SETTINGS.min_dollar_volume: continue
        rows.append({"Ticker":m["Ticker"],"Company":m["Security"],"Sector":m["GICS Sector"],
                     "liquidity":float(adv),"pre":.55*t["technical"]+.45*t["momentum"],**t})
    base=pd.DataFrame(rows)
    if base.empty: return base,market_regime()
    base=base.sort_values("pre",ascending=False).head(deep_candidates).copy()

    ext=[]
    for _,row in base.iterrows():
        t,info=get_ticker_bundle(row["Ticker"]); data={}
        if t is not None:
            for fn,args in [(fundamentals,(info,)),(valuation,(info,)),(analyst_metrics,(info,)),
                            (news_signals,(t,)),(earnings_signals,(t,)),(insider_institutional,(t,))]:
                try: data.update(fn(*args))
                except Exception: pass
        ext.append(data)
    f=pd.concat([base,pd.DataFrame(ext,index=base.index)],axis=1)

    defaults={"growth":50,"quality":50,"balance":50,"cashflow":50,"valuation":50,"analyst":50,"news_score":50,
              "earnings_score":50,"insider_score":50,"options_score":50,"forward_pe":np.nan,"fcf_yield":np.nan,
              "debt_to_equity":np.nan,"earnings_growth":np.nan,"next_earnings_days":np.nan,"news_count":0,
              "news_items":None,"atm_iv":np.nan,"put_call_oi":np.nan,"option_expiry":None}
    for c,v in defaults.items():
        if c not in f.columns: f[c]=v
        elif v is not None: f[c]=f[c].where(f[c].notna(),v)

    f=sector_relative(f)
    prelim=(.55*f["technical"]+.45*f["momentum"]).sort_values(ascending=False).head(option_candidates).index
    for i in prelim:
        t,_=get_ticker_bundle(f.at[i,"Ticker"])
        if t is None: continue
        try:
            for k,v in options_signals(t,f.at[i,"price"]).items(): f.at[i,k]=v
        except Exception: pass

    regime=market_regime()
    drift=drift_score(regime_snapshot())
    regime["drift_score"]=drift
    f=apply_scores(f,regime["score"])
    if drift<35:
        f["ShortScore"]=(f["ShortScore"]-5).clip(0,100)
    f["LongConfidence"]=f.apply(lambda r:confidence_score(r,"Long"),axis=1)
    f["ShortConfidence"]=f.apply(lambda r:confidence_score(r,"Short"),axis=1)
    return f.reset_index(drop=True),regime
