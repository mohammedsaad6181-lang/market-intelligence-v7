from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    universe_name: str = "S&P 500"
    min_dollar_volume: float = 25_000_000
    deep_candidates: int = 60
    option_candidates: int = 20
    max_sector_picks: int = 2
    min_confidence: float = 55.0
    max_data_age_days: int = 5
    risk_pct_default: float = 1.0
    stop_atr_mult: float = 2.0
    target1_atr_mult: float = 2.0
    target2_atr_mult: float = 4.0
    db_name: str = "market_intelligence.db"
    benchmark_long: str = "SPY"
    benchmark_short: str = "QQQ"

SETTINGS = Settings()
