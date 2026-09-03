import numpy as np
import pandas as pd

def metrics(returns):
    r=pd.Series(returns).dropna()
    if r.empty: return {"return_pct":0,"cagr_pct":0,"max_drawdown_pct":0,"sharpe":np.nan,"sortino":np.nan,"win_rate_pct":np.nan,"profit_factor":np.nan}
    eq=(1+r).cumprod()
    total=(eq.iloc[-1]-1)*100
    years=max(len(r)/252,1/252)
    cagr=(eq.iloc[-1]**(1/years)-1)*100 if eq.iloc[-1]>0 else -100
    dd=(eq/eq.cummax()-1).min()*100
    vol=r.std()*np.sqrt(252); sharpe=r.mean()*252/vol if vol and vol>0 else np.nan
    downside=r[r<0].std()*np.sqrt(252); sortino=r.mean()*252/downside if downside and downside>0 else np.nan
    wins=r[r>0].sum(); losses=abs(r[r<0].sum()); pf=wins/losses if losses>0 else np.nan
    return {"return_pct":total,"cagr_pct":cagr,"max_drawdown_pct":dd,"sharpe":sharpe,"sortino":sortino,
            "win_rate_pct":(r>0).mean()*100,"profit_factor":pf}

def trend_backtest(close,fast=50,slow=200,cost_bps=5):
    c=pd.Series(close).dropna()
    if len(c)<slow+30: return None
    f=c.rolling(fast).mean(); s=c.rolling(slow).mean()
    sig=(f>s).astype(int); pos=sig.shift(1).fillna(0)
    ret=c.pct_change().fillna(0); turnover=pos.diff().abs().fillna(0)
    strat=pos*ret-turnover*(cost_bps/10000)
    return {"strategy_returns":strat,"buyhold_returns":ret,"signal":sig}

def walk_forward(close,train_days=504,test_days=126,fast_grid=(20,50,100),slow_grid=(120,150,200)):
    c=pd.Series(close).dropna(); rows=[]; start=0
    while start+train_days+test_days<=len(c):
        train=c.iloc[start:start+train_days]; test=c.iloc[start+train_days:start+train_days+test_days]
        best=None
        for f in fast_grid:
            for s in slow_grid:
                if f>=s: continue
                bt=trend_backtest(train,f,s)
                if not bt: continue
                m=metrics(bt["strategy_returns"]); score=m["sharpe"] if pd.notna(m["sharpe"]) else -99
                if best is None or score>best["score"]: best={"fast":f,"slow":s,"score":score}
        if best:
            joined=pd.concat([train.tail(best["slow"]),test])
            bt=trend_backtest(joined,best["fast"],best["slow"])
            if bt:
                m=metrics(bt["strategy_returns"].tail(len(test)))
                rows.append({"start":test.index[0],"end":test.index[-1],"fast":best["fast"],"slow":best["slow"],**m})
        start+=test_days
    return pd.DataFrame(rows)

def equal_weight_portfolio(price_df,rebalance=21,cost_bps=5):
    px=price_df.dropna(how="all").ffill()
    if px.empty: return pd.Series(dtype=float)
    rets=px.pct_change().fillna(0)
    weights=pd.DataFrame(0.0,index=px.index,columns=px.columns)
    for i in range(0,len(px),rebalance):
        cols=px.iloc[i].dropna().index
        if len(cols): weights.loc[px.index[i],cols]=1/len(cols)
    weights=weights.replace(0,np.nan).ffill().fillna(0)
    turnover=weights.diff().abs().sum(axis=1).fillna(0)
    return (weights.shift(1).fillna(0)*rets).sum(axis=1)-turnover*(cost_bps/10000)
