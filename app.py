from datetime import datetime
import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from config import SETTINGS
from engine import scan
from quality import diversify_top,stale_guard
from risk import atr_levels,position_size,rr
from storage import save_picks,history,update_horizon_returns
from backtest import trend_backtest,metrics,walk_forward,equal_weight_portfolio
from ai_analyst import rules_explanation,optional_llm_analysis
from portfolio import suggested_weights,portfolio_risk_summary
from advanced_risk import beta,historical_var,historical_cvar,correlation_matrix,portfolio_volatility,monte_carlo_terminal
from quality import circuit_breaker
from migrations import migrate

st.set_page_config(page_title="Market Intelligence Pro V6",page_icon="🧠",layout="wide")
migrate()
st.title("🧠 Market Intelligence Pro V7 — Production Candidate")
st.caption("Production Candidate: Ranking + AI + Confidence + Diversification + Alpha + Portfolio Risk + Walk‑Forward + Circuit Breaker")

with st.sidebar:
    deep=st.slider("التحليل العميق",35,90,SETTINGS.deep_candidates)
    optn=st.slider("Options candidates",5,30,SETTINGS.option_candidates)
    account=st.number_input("حجم الحساب ($)",min_value=100.0,value=10000.0,step=100.0)
    risk_pct=st.number_input("المخاطرة لكل صفقة (%)",min_value=.1,max_value=5.0,value=SETTINGS.risk_pct_default,step=.1)
    max_sector=st.slider("أقصى عدد من نفس القطاع في Top 5",1,5,SETTINGS.max_sector_picks)
    min_conf=st.slider("أقل Confidence",0,100,int(SETTINGS.min_confidence))
    use_llm=st.toggle("AI لغوي اختياري",False)

if st.button("🚀 تشغيل V6",type="primary",use_container_width=True):
    with st.spinner("جاري الفحص..."):
        df,regime=scan(deep,optn)
        st.session_state["df"]=df; st.session_state["regime"]=regime

df=st.session_state.get("df")
regime=st.session_state.get("regime",{"score":50,"label":"Unknown","vix":np.nan})
m1,m2,m3,m4=st.columns(4)
m1.metric("Market",regime.get("label","Unknown")); m2.metric("Score",f"{regime.get('score',50)}/100")
vix=regime.get("vix",np.nan); m3.metric("VIX",f"{vix:.1f}" if pd.notna(vix) else "N/A")
m4.metric("AI","LLM + Rules" if use_llm and os.getenv("OPENAI_API_KEY") else "Rules")

if df is None:
    st.info("شغّل الفحص أولًا."); st.stop()
ok,msg,health=circuit_breaker(df,min_health=75,min_rows=20)
if not ok:
    st.error(f"Circuit Breaker: {msg}"); st.stop()
st.success(f"Data Health: {health:.1f}/100")

long5=diversify_top(df,"LongScore",5,max_sector,min_conf)
short5=diversify_top(df,"ShortScore",5,max_sector,min_conf)
long5["Rank"]=range(1,len(long5)+1); short5["Rank"]=range(1,len(short5)+1)

tabs=st.tabs(["🏦 Long","⚡ Short","🧠 AI","🎯 Confidence","🛡️ Risk","🧺 Portfolio Lab","🧪 Backtest","🧭 Walk‑Forward","📈 Alpha Tracker","🧮 Advanced Risk","🩺 Diagnostics"])

with tabs[0]:
    cols=["Rank","Ticker","Company","Sector","LongScore","LongConfidence","growth","quality","valuation","sector_value_score","cashflow","balance","price"]
    out=long5[cols].copy()
    out.columns=["#","Ticker","الشركة","القطاع","Long Score","Confidence","Growth","Quality","Valuation","Sector Value","Cash Flow","Balance","السعر"]
    st.dataframe(out.round(2),use_container_width=True,hide_index=True)
    if st.button("💾 حفظ Long"):
        save_picks(long5,"Long"); st.success("تم الحفظ")

with tabs[1]:
    cols=["Rank","Ticker","Company","Sector","ShortScore","ShortConfidence","technical","momentum","news_score","earnings_score","options_score","risk","price"]
    out=short5[cols].copy()
    out.columns=["#","Ticker","الشركة","القطاع","Short Score","Confidence","Technical","Momentum","News","Earnings","Options","Risk","السعر"]
    st.dataframe(out.round(2),use_container_width=True,hide_index=True)
    if st.button("💾 حفظ Short"):
        save_picks(short5,"Short"); st.success("تم الحفظ")

with tabs[2]:
    ticks=list(dict.fromkeys(long5["Ticker"].tolist()+short5["Ticker"].tolist()))
    t=st.selectbox("السهم",ticks,key="ai"); horizon=st.radio("الأفق",["Long","Short"],horizontal=True)
    r=df[df["Ticker"]==t].iloc[0].to_dict(); rule=rules_explanation(r,horizon,regime)
    st.write(rule)
    if use_llm:
        if os.getenv("OPENAI_API_KEY"):
            if st.button("✨ تحليل AI"):
                st.session_state["ai_text"]=optional_llm_analysis(r,horizon,rule)
            if st.session_state.get("ai_text"): st.write(st.session_state["ai_text"])
        else: st.warning("OPENAI_API_KEY غير موجود")

with tabs[3]:
    comp=pd.DataFrame({
        "Ticker":df["Ticker"],"Long Score":df["LongScore"],"Long Confidence":df["LongConfidence"],
        "Short Score":df["ShortScore"],"Short Confidence":df["ShortConfidence"],"Sector":df["Sector"]
    }).sort_values("Long Score",ascending=False).head(30)
    st.dataframe(comp.round(2),use_container_width=True,hide_index=True)
    st.info("Confidence يقيس اكتمال واتساق البيانات، وليس احتمال الربح.")

with tabs[4]:
    t=st.selectbox("السهم",list(dict.fromkeys(short5["Ticker"].tolist()+long5["Ticker"].tolist())),key="risk")
    r=df[df["Ticker"]==t].iloc[0]
    stop,t1,t2=atr_levels(r["price"],r["atr"])
    ps=position_size(account,risk_pct,r["price"],stop)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Entry",f"${r['price']:.2f}"); c2.metric("Stop",f"${stop:.2f}")
    c3.metric("Shares",str(ps["shares"])); c4.metric("Risk $",f"${ps['risk_amount']:.2f}")
    st.write(f"Position value: **${ps['position_value']:.2f}** | T1 **${t1:.2f}** | T2 **${t2:.2f}** | R/R **{rr(r['price'],stop,t2):.2f}:1**")
    if ps["capped"]: st.warning("تم تقليل الحجم بسبب حد أقصى 25% من الحساب في مركز واحد.")

with tabs[5]:
    horizon=st.radio("المحفظة",["Long","Short"],horizontal=True,key="port")
    picks=long5.copy() if horizon=="Long" else short5.copy()
    sc="LongScore" if horizon=="Long" else "ShortScore"; cc="LongConfidence" if horizon=="Long" else "ShortConfidence"
    weighted=suggested_weights(picks,sc,cc)
    st.dataframe(weighted[["Ticker","Sector",sc,cc,"Weight"]].assign(Weight=lambda x:(x["Weight"]*100).round(2)),use_container_width=True,hide_index=True)
    summ=portfolio_risk_summary(weighted)
    st.write(f"عدد القطاعات: **{summ.get('sector_count',0)}** | أكبر تركّز قطاعي: **{summ.get('largest_sector_share',0):.1f}%**")

with tabs[6]:
    t=st.selectbox("السهم",list(dict.fromkeys(short5["Ticker"].tolist()+long5["Ticker"].tolist())),key="bt")
    h=yf.download(t,period="10y",auto_adjust=True,progress=False)
    if not h.empty:
        c=h["Close"]; c=c.iloc[:,0] if isinstance(c,pd.DataFrame) else c
        bt=trend_backtest(c)
        if bt:
            tbl=pd.DataFrame([{"Model":"Strategy",**metrics(bt["strategy_returns"])},{"Model":"Buy&Hold",**metrics(bt["buyhold_returns"])}])
            st.dataframe(tbl.round(2),use_container_width=True,hide_index=True)
            st.line_chart(pd.DataFrame({"Strategy":(1+bt["strategy_returns"]).cumprod(),"BuyHold":(1+bt["buyhold_returns"]).cumprod()}))

with tabs[7]:
    t=st.selectbox("السهم",list(dict.fromkeys(short5["Ticker"].tolist()+long5["Ticker"].tolist())),key="wf")
    h=yf.download(t,period="10y",auto_adjust=True,progress=False)
    if not h.empty:
        c=h["Close"]; c=c.iloc[:,0] if isinstance(c,pd.DataFrame) else c
        wf=walk_forward(c)
        st.dataframe(wf.round(2),use_container_width=True,hide_index=True)
        if not wf.empty:
            st.write(f"Mean OOS Sharpe: **{wf['sharpe'].mean():.2f}** | Mean OOS return/window: **{wf['return_pct'].mean():.2f}%**")

with tabs[8]:
    if st.button("🔄 تحديث 1D/5D/21D/63D + Alpha"):
        st.session_state["hist"]=update_horizon_returns()
    hist=st.session_state.get("hist")
    if hist is None: hist=history()
    if hist.empty:
        st.info("لا يوجد سجل بعد.")
    else:
        st.dataframe(hist.round(2),use_container_width=True,hide_index=True)
        cols=[c for c in ["alpha_1d","alpha_5d","alpha_21d","alpha_63d"] if c in hist.columns]
        if cols:
            summary=hist.groupby("horizon")[cols].mean().reset_index()
            st.subheader("متوسط Alpha مقابل Benchmark")
            st.dataframe(summary.round(2),use_container_width=True,hide_index=True)

with tabs[9]:
    picks=long5 if len(long5)>=2 else short5
    tickers=picks["Ticker"].tolist()
    if tickers:
        px=yf.download(tickers+["SPY"],period="1y",auto_adjust=True,progress=False)
        try:
            closes=px["Close"] if isinstance(px.columns,pd.MultiIndex) else px[["Close"]]
            if isinstance(closes,pd.Series): closes=closes.to_frame()
            stock_cols=[c for c in tickers if c in closes.columns]
            if stock_cols:
                corr=correlation_matrix(closes[stock_cols])
                st.subheader("Correlation Matrix")
                st.dataframe(corr.round(2),use_container_width=True)
                st.write(f"Equal-weight annualized portfolio volatility: **{portfolio_volatility(closes[stock_cols]):.2f}%**")
                t=st.selectbox("Risk ticker",stock_cols,key="adv_risk")
                sr=closes[t].pct_change().dropna()
                br=closes["SPY"].pct_change().dropna() if "SPY" in closes.columns else pd.Series(dtype=float)
                c1,c2,c3=st.columns(3)
                c1.metric("Beta vs SPY",f"{beta(sr,br):.2f}" if len(br) else "N/A")
                c2.metric("95% VaR (1D)",f"{historical_var(sr):.2f}%")
                c3.metric("95% CVaR (1D)",f"{historical_cvar(sr):.2f}%")
                mc=monte_carlo_terminal(sr)
                if mc:
                    st.write(f"Monte Carlo 1Y median: **{mc['median_return_pct']:.1f}%** | P10: **{mc['p10_return_pct']:.1f}%** | P90: **{mc['p90_return_pct']:.1f}%** | Probability of loss: **{mc['prob_loss_pct']:.1f}%**")
                    st.caption("Monte Carlo is a statistical scenario model, not a forecast.")
        except Exception as e:
            st.warning(f"Advanced risk data unavailable: {e}")

with tabs[10]:
    missing=(df.isna().mean()*100).sort_values(ascending=False).head(20).reset_index()
    missing.columns=["Field","Missing %"]
    st.dataframe(missing.round(2),use_container_width=True,hide_index=True)
    st.write(f"Stocks deep-analyzed: **{len(df)}**")
    st.write(f"Long confidence avg: **{df['LongConfidence'].mean():.1f}** | Short confidence avg: **{df['ShortConfidence'].mean():.1f}**")
    st.info("أي حقل بجودة ضعيفة يُعامل بحذر، وConfidence لا يمثل احتمال النجاح.")

st.download_button("⬇️ CSV",df.to_csv(index=False).encode("utf-8-sig"),file_name=f"market_v6_{datetime.now():%Y%m%d}.csv",mime="text/csv")
st.caption("V7 Production Candidate. لا توجد استراتيجية مضمونة، وVaR/Monte Carlo/Backtests ليست تنبؤًا بالمستقبل.")
