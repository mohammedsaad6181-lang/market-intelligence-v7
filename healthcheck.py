import sqlite3
from pathlib import Path
from config import SETTINGS
from migrations import migrate

def main():
    required=["app.py","engine.py","storage.py","indicators.py","fundamentals.py","advanced_risk.py","quality.py","daily_runner.py"]
    missing=[f for f in required if not Path(__file__).with_name(f).exists()]
    if missing:
        print("Missing:",missing); return 1
    try:
        migrate()
        con=sqlite3.connect(Path(__file__).with_name(SETTINGS.db_name))
        con.execute("SELECT 1")
        cols={r[1] for r in con.execute("PRAGMA table_info(picks)").fetchall()}
        for c in ["model_version","regime_score","data_confidence"]:
            if c not in cols:
                print("Migration missing:",c); return 3
        con.close()
    except Exception as e:
        print("DB failed:",e); return 2
    print("healthcheck: OK")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
