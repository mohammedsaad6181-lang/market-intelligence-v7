import numpy as np
import pandas as pd
from indicators import norm
from data_provider import safe_float

def pct(x): return x*100 if pd.notna(x) else np.nan

def fundamentals(info):
    rg=pct(safe_float(info.get("revenueGrowth"))); eg=pct(safe_float(info.get("earningsGrowth")))
    pm=pct(safe_float(info.get("profitMargins"))); roe=pct(safe_float(info.get("returnOnEquity")))
    gm=pct(safe_float(info.get("grossMargins"))); om=pct(safe_float(info.get("operatingMargins")))
    de=safe_float(info.get("debtToEquity")); cr=safe_float(info.get("currentRatio"))
    fcf=safe_float(info.get("freeCashflow")); mcap=safe_float(info.get("marketCap"))
    fcfy=fcf/mcap*100 if pd.notna(fcf) and pd.notna(mcap) and mcap>0 else np.nan
    return {
        "growth":.45*norm(rg,-5,30)+.55*norm(eg,-10,45),
        "quality":.25*norm(pm,2,30)+.25*norm(roe,5,40)+.25*norm(gm,10,65)+.25*norm(om,3,35),
        "balance":.55*norm(de,40,250,invert=True)+.45*norm(cr,.7,2.5),
        "cashflow":norm(fcfy,0,8),
        "revenue_growth":rg,"earnings_growth":eg,"profit_margin":pm,"roe":roe,
        "gross_margin":gm,"operating_margin":om,"debt_to_equity":de,"current_ratio":cr,"fcf_yield":fcfy
    }

def valuation(info):
    pe=safe_float(info.get("trailingPE")); fpe=safe_float(info.get("forwardPE"))
    peg=safe_float(info.get("pegRatio")); ps=safe_float(info.get("priceToSalesTrailing12Months"))
    pb=safe_float(info.get("priceToBook")); eveb=safe_float(info.get("enterpriseToEbitda"))
    vals=[norm(fpe if pd.notna(fpe) else pe,12,45,invert=True),norm(peg,.8,3,invert=True),
          norm(ps,1,12,invert=True),norm(eveb,8,35,invert=True)]
    return {"valuation":float(np.nanmean(vals)),"pe":pe,"forward_pe":fpe,"peg":peg,
            "price_sales":ps,"price_book":pb,"ev_ebitda":eveb}

def analyst_metrics(info):
    target=safe_float(info.get("targetMeanPrice")); price=safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    upside=(target/price-1)*100 if pd.notna(target) and pd.notna(price) and price>0 else np.nan
    n=safe_float(info.get("numberOfAnalystOpinions")); rec=safe_float(info.get("recommendationMean"))
    score=.65*norm(upside,-15,35)+.20*norm(n,3,35)+.15*norm(rec,1,4,invert=True)
    return {"analyst":score,"analyst_upside":upside,"analyst_count":n,"recommendation_mean":rec,"target_mean":target}
