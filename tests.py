import pandas as pd
import numpy as np
from risk import atr_levels,position_size,rr
from backtest import metrics
from indicators import norm
from quality import diversify_top,confidence_score,data_health_score,circuit_breaker
from advanced_risk import beta,historical_var,historical_cvar,correlation_matrix,portfolio_volatility,monte_carlo_terminal

def run():
    assert norm(0,0,10)==0 and norm(10,0,10)==100
    assert atr_levels(100,5)==(90,110,120)
    p=position_size(10000,1,100,95)
    assert p["shares"]==20
    assert round(rr(100,95,110),2)==2.0

    m=metrics(pd.Series([.01,-.005,.002,.003,-.001]))
    assert "profit_factor" in m and "sortino" in m

    df=pd.DataFrame([
        {"Ticker":"A","Sector":"Tech","LongScore":90,"LongConfidence":80,"price":100,"technical":80,"momentum":80,"risk":70,"growth":80,"quality":80,"valuation":70,"earnings_score":70},
        {"Ticker":"B","Sector":"Tech","LongScore":89,"LongConfidence":80,"price":101,"technical":80,"momentum":80,"risk":70,"growth":80,"quality":80,"valuation":70,"earnings_score":70},
        {"Ticker":"C","Sector":"Tech","LongScore":88,"LongConfidence":80,"price":102,"technical":80,"momentum":80,"risk":70,"growth":80,"quality":80,"valuation":70,"earnings_score":70},
        {"Ticker":"D","Sector":"Health","LongScore":87,"LongConfidence":80,"price":103,"technical":80,"momentum":80,"risk":70,"growth":80,"quality":80,"valuation":70,"earnings_score":70},
        {"Ticker":"E","Sector":"Energy","LongScore":86,"LongConfidence":80,"price":104,"technical":80,"momentum":80,"risk":70,"growth":80,"quality":80,"valuation":70,"earnings_score":70},
        {"Ticker":"F","Sector":"Finance","LongScore":85,"LongConfidence":80,"price":105,"technical":80,"momentum":80,"risk":70,"growth":80,"quality":80,"valuation":70,"earnings_score":70},
    ])
    top=diversify_top(df,"LongScore",5,2,55)
    assert len(top)==5 and top["Sector"].value_counts().max()<=2
    assert data_health_score(df)>90
    ok,_,_=circuit_breaker(df,min_health=50,min_rows=5)
    assert ok

    x=pd.Series([.01,.02,-.01,.005,.012,-.004]*20)
    y=pd.Series([.008,.015,-.008,.004,.01,-.003]*20)
    assert pd.notna(beta(x,y))
    assert historical_var(x)>0
    assert historical_cvar(x)>0

    prices=pd.DataFrame({"A":100*(1+x).cumprod(),"B":100*(1+y).cumprod()})
    corr=correlation_matrix(prices)
    assert corr.shape==(2,2)
    assert portfolio_volatility(prices)>0
    mc=monte_carlo_terminal(x,days=30,simulations=500)
    assert "prob_loss_pct" in mc

    print("All V7 production core tests passed.")

if __name__=="__main__":
    run()
