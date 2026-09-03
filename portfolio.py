import numpy as np
import pandas as pd

def suggested_weights(df,score_col,confidence_col,max_weight=0.25):
    x=df.copy()
    raw=(x[score_col].clip(lower=1)*x[confidence_col].clip(lower=1))
    if raw.sum()<=0:
        x["Weight"]=1/len(x)
        return x
    w=raw/raw.sum()
    w=w.clip(upper=max_weight)
    w=w/w.sum()
    x["Weight"]=w
    return x

def portfolio_risk_summary(df):
    if df.empty: return {}
    return {
        "avg_score":float(df.filter(regex="Score$").max(axis=1).mean()),
        "avg_confidence":float(df.filter(regex="Confidence$").max(axis=1).mean()),
        "sector_count":int(df["Sector"].nunique()),
        "largest_sector_share":float(df["Sector"].value_counts(normalize=True).max()*100)
    }
