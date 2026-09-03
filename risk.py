import math

def atr_levels(price,atr,stop_mult=2,t1_mult=2,t2_mult=4):
    return price-stop_mult*atr,price+t1_mult*atr,price+t2_mult*atr

def position_size(account_size,risk_pct,entry,stop,max_position_pct=25):
    if account_size<=0 or risk_pct<=0 or entry<=0 or stop<=0 or entry<=stop:
        return {"shares":0,"risk_amount":0.0,"position_value":0.0,"capped":False}
    risk_amount=account_size*(risk_pct/100)
    rps=entry-stop
    shares=max(0,math.floor(risk_amount/rps))
    max_value=account_size*(max_position_pct/100)
    cap_shares=max(0,math.floor(max_value/entry))
    capped=shares>cap_shares
    shares=min(shares,cap_shares)
    return {"shares":shares,"risk_amount":shares*rps,"position_value":shares*entry,"capped":capped}

def rr(entry,stop,target):
    if entry<=stop: return None
    return (target-entry)/(entry-stop)
