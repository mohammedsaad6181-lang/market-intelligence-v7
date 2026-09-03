import sqlite3
from pathlib import Path
from datetime import date,datetime,timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from config import SETTINGS

DB_PATH=Path(__file__).with_name(SETTINGS.db_name)

def connect():
    con=sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
    CREATE TABLE IF NOT EXISTS picks(
      pick_date TEXT NOT NULL,horizon TEXT NOT NULL,rank INTEGER NOT NULL,ticker TEXT NOT NULL,sector TEXT,
      score REAL NOT NULL,confidence REAL,entry_price REAL NOT NULL,price_now REAL,return_pct REAL,
      ret_1d REAL,ret_5d REAL,ret_21d REAL,ret_63d REAL,
      bench_1d REAL,bench_5d REAL,bench_21d REAL,bench_63d REAL,
      alpha_1d REAL,alpha_5d REAL,alpha_21d REAL,alpha_63d REAL,
      updated_at TEXT,PRIMARY KEY(pick_date,horizon,ticker)
    )""")
    con.commit()
    return con

def save_picks(df,horizon,date_str=None):
    if date_str is None: date_str=date.today().isoformat()
    score_col="LongScore" if horizon=="Long" else "ShortScore"
    conf_col="LongConfidence" if horizon=="Long" else "ShortConfidence"
    con=connect()
    for rank,(_,r) in enumerate(df.iterrows(),1):
        con.execute("""INSERT OR REPLACE INTO picks
        (pick_date,horizon,rank,ticker,sector,score,confidence,entry_price,price_now,return_pct,updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        (date_str,horizon,rank,r["Ticker"],r.get("Sector"),float(r[score_col]),float(r.get(conf_col,0)),
         float(r["price"]),float(r["price"]),0.0))
    con.commit(); con.close()

def history():
    con=connect(); df=pd.read_sql_query("SELECT * FROM picks ORDER BY pick_date DESC,horizon,rank",con); con.close(); return df

def _price_return(ticker,start_date,horizon_days):
    start=pd.Timestamp(start_date)
    end=start+pd.Timedelta(days=max(horizon_days*2+10,20))
    try:
        d=yf.download(ticker,start=start.strftime("%Y-%m-%d"),end=end.strftime("%Y-%m-%d"),auto_adjust=True,progress=False)
        if d.empty: return np.nan
        c=d["Close"]
        if isinstance(c,pd.DataFrame): c=c.iloc[:,0]
        c=c.dropna()
        if len(c)<2: return np.nan
        idx=min(horizon_days,len(c)-1)
        return float((c.iloc[idx]/c.iloc[0]-1)*100)
    except Exception: return np.nan

def update_horizon_returns():
    con=connect()
    df=pd.read_sql_query("SELECT * FROM picks",con)
    today=pd.Timestamp.today().normalize()
    for _,r in df.iterrows():
        pick=pd.Timestamp(r["pick_date"])
        age=(today-pick).days
        bench=SETTINGS.benchmark_long if r["horizon"]=="Long" else SETTINGS.benchmark_short
        updates={}
        for days,col in [(1,"1d"),(5,"5d"),(21,"21d"),(63,"63d")]:
            if age>=days and pd.isna(r.get(f"ret_{col}",np.nan)):
                sr=_price_return(r["ticker"],r["pick_date"],days)
                br=_price_return(bench,r["pick_date"],days)
                updates[f"ret_{col}"]=sr; updates[f"bench_{col}"]=br
                updates[f"alpha_{col}"]=sr-br if pd.notna(sr) and pd.notna(br) else np.nan
        if updates:
            sets=",".join([f"{k}=?" for k in updates])+",updated_at=datetime('now')"
            vals=list(updates.values())+[r["pick_date"],r["horizon"],r["ticker"]]
            con.execute(f"UPDATE picks SET {sets} WHERE pick_date=? AND horizon=? AND ticker=?",vals)
    con.commit()
    out=pd.read_sql_query("SELECT * FROM picks ORDER BY pick_date DESC,horizon,rank",con)
    con.close(); return out
