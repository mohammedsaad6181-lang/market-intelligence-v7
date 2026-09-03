import numpy as np
import pandas as pd

def beta(stock_returns, benchmark_returns):
    df=pd.concat([pd.Series(stock_returns),pd.Series(benchmark_returns)],axis=1).dropna()
    if len(df)<30: return np.nan
    s,b=df.iloc[:,0],df.iloc[:,1]
    var=b.var()
    if var==0 or pd.isna(var): return np.nan
    return float(s.cov(b)/var)

def historical_var(returns, confidence=0.95):
    r=pd.Series(returns).dropna()
    if len(r)<30: return np.nan
    return float(-np.quantile(r,1-confidence)*100)

def historical_cvar(returns, confidence=0.95):
    r=pd.Series(returns).dropna()
    if len(r)<30: return np.nan
    cutoff=np.quantile(r,1-confidence)
    tail=r[r<=cutoff]
    return float(-tail.mean()*100) if len(tail) else np.nan

def correlation_matrix(price_df):
    if price_df is None or price_df.empty: return pd.DataFrame()
    return price_df.pct_change().corr()

def portfolio_volatility(price_df, weights=None):
    if price_df is None or price_df.empty: return np.nan
    r=price_df.pct_change().dropna(how="all")
    if r.empty: return np.nan
    cols=r.columns
    if weights is None:
        w=np.repeat(1/len(cols),len(cols))
    else:
        w=np.array([weights.get(c,0) for c in cols],dtype=float)
        if w.sum()<=0: return np.nan
        w=w/w.sum()
    cov=r.cov().values*252
    return float(np.sqrt(w.T@cov@w)*100)

def monte_carlo_terminal(returns, days=252, simulations=5000, seed=42):
    r=pd.Series(returns).dropna()
    if len(r)<30: return {}
    mu=float(r.mean()); sigma=float(r.std())
    rng=np.random.default_rng(seed)
    sims=rng.normal(mu,sigma,size=(simulations,days))
    terminal=np.prod(1+sims,axis=1)
    return {
        "median_return_pct":float((np.median(terminal)-1)*100),
        "p10_return_pct":float((np.quantile(terminal,.10)-1)*100),
        "p90_return_pct":float((np.quantile(terminal,.90)-1)*100),
        "prob_loss_pct":float((terminal<1).mean()*100),
    }
