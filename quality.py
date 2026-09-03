import numpy as np
import pandas as pd

CRITICAL_LONG=["growth","quality","valuation","cashflow","balance","earnings_score","price"]
CRITICAL_SHORT=["technical","momentum","risk","news_score","earnings_score","price","rsi"]

def confidence_score(row,horizon):
    fields=CRITICAL_LONG if horizon=="Long" else CRITICAL_SHORT
    present=sum(pd.notna(row.get(f,np.nan)) for f in fields)
    completeness=present/len(fields)*100
    consistency=50
    if horizon=="Long":
        vals=[row.get("growth"),row.get("quality"),row.get("cashflow"),row.get("balance")]
    else:
        vals=[row.get("technical"),row.get("momentum"),row.get("risk")]
    vals=[float(v) for v in vals if pd.notna(v)]
    if vals:
        spread=np.std(vals)
        consistency=max(0,100-spread*2)
    return float(.72*completeness+.28*consistency)

def stale_guard(df):
    if df is None or df.empty: return False,"No data"
    if "price" not in df.columns or df["price"].isna().all(): return False,"Missing prices"
    return True,"OK"

def diversify_top(df,score_col,n=5,max_per_sector=2,min_confidence=55):
    conf_col="LongConfidence" if score_col=="LongScore" else "ShortConfidence"
    ranked=df.sort_values(score_col,ascending=False)
    chosen=[]; sector_counts={}
    for _,r in ranked.iterrows():
        if float(r.get(conf_col,0))<min_confidence: continue
        sector=r.get("Sector","Unknown")
        if sector_counts.get(sector,0)>=max_per_sector: continue
        chosen.append(r)
        sector_counts[sector]=sector_counts.get(sector,0)+1
        if len(chosen)>=n: break
    if len(chosen)<n:
        used={r["Ticker"] for r in chosen}
        for _,r in ranked.iterrows():
            if r["Ticker"] in used: continue
            chosen.append(r)
            if len(chosen)>=n: break
    return pd.DataFrame(chosen).reset_index(drop=True)


def data_health_score(df):
    if df is None or df.empty: return 0.0
    critical=["price","technical","momentum","risk","growth","quality","valuation","earnings_score"]
    existing=[c for c in critical if c in df.columns]
    if not existing: return 0.0
    completeness=1-df[existing].isna().mean().mean()
    sane=1.0
    if "price" in df.columns:
        sane*=float((pd.to_numeric(df["price"],errors="coerce")>0).mean())
    return float(max(0,min(100,completeness*sane*100)))

def circuit_breaker(df,min_health=75,min_rows=20):
    health=data_health_score(df)
    if df is None or len(df)<min_rows:
        return False,f"Insufficient rows: {0 if df is None else len(df)}",health
    if health<min_health:
        return False,f"Data health too low: {health:.1f}",health
    return True,"OK",health
