import numpy as np
import pandas as pd
from indicators import norm
from data_provider import safe_float

POS={"beat","record","growth","surge","strong","raises","upgrade","profit","expands","approval","wins","launch","partnership","outperform"}
NEG={"miss","cuts","downgrade","lawsuit","probe","investigation","decline","weak","warning","recall","loss","layoffs","delay","fraud"}

def _df_attr(t,name):
    try:
        v=getattr(t,name)
        if callable(v): v=v()
        return v if isinstance(v,pd.DataFrame) else pd.DataFrame()
    except Exception: return pd.DataFrame()

def news_signals(t,max_items=12):
    try: raw=t.news or []
    except Exception: raw=[]
    items=[]
    for n in raw[:max_items]:
        if not isinstance(n,dict): continue
        content=n.get("content") if isinstance(n.get("content"),dict) else {}
        title=n.get("title") or content.get("title") or ""
        publisher=n.get("publisher") or (content.get("provider") or {}).get("displayName","")
        txt=title.lower()
        s=sum(w in txt for w in POS)-sum(w in txt for w in NEG)
        items.append({"title":title,"publisher":publisher,"raw_score":s})
    if not items: return {"news_score":50.0,"news_sentiment":0.0,"news_count":0,"news_items":[]}
    sent=float(np.mean([i["raw_score"] for i in items]))
    return {"news_score":norm(sent,-1.5,1.5),"news_sentiment":sent,"news_count":len(items),"news_items":items}

def earnings_signals(t):
    surprise=np.nan; revisions=np.nan; days=np.nan
    hist=_df_attr(t,"earnings_history")
    try:
        if not hist.empty:
            for c in hist.columns:
                if "surprise" in str(c).lower():
                    vals=pd.to_numeric(hist[c],errors="coerce").dropna()
                    if len(vals): surprise=float(vals.tail(4).mean())
                    break
    except Exception: pass
    rev=_df_attr(t,"eps_revisions")
    try:
        if not rev.empty: revisions=float(rev.apply(pd.to_numeric,errors="coerce").mean(axis=1,skipna=True).mean())
    except Exception: pass
    try:
        cal=t.calendar
        if isinstance(cal,dict):
            ed=cal.get("Earnings Date")
            if isinstance(ed,(list,tuple)) and ed:
                dt=pd.to_datetime(ed[0],errors="coerce")
                if pd.notna(dt):
                    if getattr(dt,"tzinfo",None): dt=dt.tz_localize(None)
                    days=(dt-pd.Timestamp.now().normalize()).days
    except Exception: pass
    return {"earnings_score":.6*norm(surprise,-10,15)+.4*norm(revisions,-1,1),
            "earnings_surprise_avg":surprise,"eps_revisions_proxy":revisions,"next_earnings_days":days}

def insider_institutional(t):
    score=50.0; net=np.nan; institutional=np.nan
    ins=_df_attr(t,"insider_transactions")
    try:
        if not ins.empty:
            text=ins.astype(str).apply(lambda col: col.str.lower())
            rows=text.apply(lambda r:" ".join(r.values),axis=1)
            buys=rows.str.contains("buy|purchase|acquisition",regex=True).sum()
            sells=rows.str.contains("sale|sell|disposition",regex=True).sum()
            net=float(buys-sells); score=norm(net,-6,6)
    except Exception: pass
    inst=_df_attr(t,"institutional_holders")
    try:
        if not inst.empty and "% Out" in inst.columns:
            institutional=float(pd.to_numeric(inst["% Out"],errors="coerce").sum()*100)
    except Exception: pass
    return {"insider_score":score,"insider_net_proxy":net,"institutional_pct_proxy":institutional}

def options_signals(t,spot):
    out={"options_score":50.0,"atm_iv":np.nan,"put_call_oi":np.nan,"option_expiry":None}
    try:
        exps=t.options
        if not exps: return out
        exp=exps[0]; ch=t.option_chain(exp); calls,puts=ch.calls.copy(),ch.puts.copy()
        if calls.empty or puts.empty: return out
        calls["dist"]=(calls["strike"]-spot).abs(); puts["dist"]=(puts["strike"]-spot).abs()
        civ=safe_float(calls.sort_values("dist").iloc[0].get("impliedVolatility"))
        piv=safe_float(puts.sort_values("dist").iloc[0].get("impliedVolatility"))
        iv=np.nanmean([civ,piv])*100
        coi=pd.to_numeric(calls.get("openInterest"),errors="coerce").sum()
        poi=pd.to_numeric(puts.get("openInterest"),errors="coerce").sum()
        pcr=poi/coi if coi and coi>0 else np.nan
        ivs=100-norm(iv,25,100); pcs=85 if pd.notna(pcr) and .55<=pcr<=1.25 else (55 if pd.notna(pcr) and .35<=pcr<=1.8 else 35)
        return {"options_score":.65*ivs+.35*pcs,"atm_iv":iv,"put_call_oi":pcr,"option_expiry":exp}
    except Exception: return out
