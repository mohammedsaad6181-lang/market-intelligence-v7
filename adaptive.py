import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from config import SETTINGS

DB_PATH=Path(__file__).with_name(SETTINGS.db_name)

BASE_LONG={
    "growth":.18,"quality":.17,"valuation":.13,"sector_value_score":.10,"cashflow":.10,
    "balance":.08,"earnings_score":.07,"analyst":.05,"insider_score":.04,"news_score":.03,
    "risk":.03,"technical":.02
}
BASE_SHORT={
    "technical":.28,"momentum":.24,"risk":.12,"news_score":.08,"earnings_score":.07,
    "options_score":.06,"analyst":.06,"growth":.04,"valuation":.03
}

def performance_multiplier(horizon):
    try:
        con=sqlite3.connect(DB_PATH)
        df=pd.read_sql_query("SELECT * FROM picks WHERE horizon=?",con,params=(horizon,))
        con.close()
    except Exception:
        return 1.0
    if len(df)<20:
        return 1.0
    cols=[c for c in ["alpha_5d","alpha_21d","alpha_63d"] if c in df.columns]
    vals=pd.concat([pd.to_numeric(df[c],errors="coerce") for c in cols],ignore_index=True).dropna()
    if len(vals)<20: return 1.0
    mean=float(vals.mean())
    return float(np.clip(1+mean/100,0.85,1.15))

def adaptive_blend(base_weights,horizon):
    # Conservative global scaling only; avoids overfitting individual factors without factor-attribution history.
    m=performance_multiplier(horizon)
    return {k:v for k,v in base_weights.items()},m
