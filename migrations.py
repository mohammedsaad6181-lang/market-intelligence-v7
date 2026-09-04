import sqlite3
from pathlib import Path
from config import SETTINGS

DB_PATH=Path(__file__).with_name(SETTINGS.db_name)

COLUMNS={
    "picks":{
        "model_version":"TEXT",
        "regime_score":"REAL",
        "data_confidence":"REAL"
    }
}

def ensure_base_schema(con):
    con.execute("""
    CREATE TABLE IF NOT EXISTS picks(
      pick_date TEXT NOT NULL,
      horizon TEXT NOT NULL,
      rank INTEGER NOT NULL,
      ticker TEXT NOT NULL,
      sector TEXT,
      score REAL NOT NULL,
      confidence REAL,
      entry_price REAL NOT NULL,
      price_now REAL,
      return_pct REAL,
      ret_1d REAL,
      ret_5d REAL,
      ret_21d REAL,
      ret_63d REAL,
      bench_1d REAL,
      bench_5d REAL,
      bench_21d REAL,
      bench_63d REAL,
      alpha_1d REAL,
      alpha_5d REAL,
      alpha_21d REAL,
      alpha_63d REAL,
      updated_at TEXT,
      PRIMARY KEY(pick_date,horizon,ticker)
    )
    """)

def column_names(con,table):
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}

def migrate():
    con = sqlite3.connect(DB_PATH, timeout=30)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=30000")
        con.execute("BEGIN IMMEDIATE")
        ensure_base_schema(con)

        for table, cols in COLUMNS.items():
            for col, typ in cols.items():
                if col in column_names(con, table):
                    continue
                try:
                    con.execute(
                        f'ALTER TABLE "{table}" ADD COLUMN "{col}" {typ}'
                    )
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise

        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

if __name__=="__main__":
    migrate()
    print("migrations: OK")
