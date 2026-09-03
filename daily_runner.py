from datetime import date
from engine import scan
from quality import diversify_top,circuit_breaker
from storage import save_picks
from config import SETTINGS
from migrations import migrate

def main():
    migrate()
    df,regime=scan()
    ok,msg,health=circuit_breaker(df,min_health=75,min_rows=20)
    if not ok:
        raise SystemExit(f"CIRCUIT BREAKER: {msg}")
    long5=diversify_top(df,"LongScore",5,SETTINGS.max_sector_picks,SETTINGS.min_confidence)
    short5=diversify_top(df,"ShortScore",5,SETTINGS.max_sector_picks,SETTINGS.min_confidence)
    if len(long5)<5 or len(short5)<5:
        raise SystemExit("CIRCUIT BREAKER: not enough diversified high-confidence candidates")
    save_picks(long5,"Long")
    save_picks(short5,"Short")
    print(f"{date.today()} | regime={regime.get('label')} | drift={regime.get('drift_score')} | health={health:.1f}")
    print("LONG:",",".join(long5["Ticker"]))
    print("SHORT:",",".join(short5["Ticker"]))

if __name__=="__main__":
    main()
