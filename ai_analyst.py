import json,os

def rules_explanation(r,horizon,regime):
    positives=[]; risks=[]
    if horizon=="Long":
        for k,thr,label in [("growth",70,"نمو قوي"),("quality",70,"جودة مرتفعة"),("cashflow",65,"تدفق نقدي جيد"),
                            ("sector_value_score",65,"تقييم نسبي جيد"),("earnings_score",65,"صورة أرباح إيجابية")]:
            try:
                if float(r.get(k,0))>=thr: positives.append(label)
            except Exception: pass
        try:
            if float(r.get("valuation",50))<40: risks.append("تقييم مرتفع")
            if float(r.get("balance",50))<40: risks.append("ميزانية أضعف من المطلوب")
        except Exception: pass
    else:
        try:
            if float(r.get("technical",0))>=70: positives.append("اتجاه فني قوي")
            if float(r.get("momentum",0))>=70: positives.append("زخم قوي")
            if 50<=float(r.get("rsi",0))<=68: positives.append("RSI صحي")
            if float(r.get("rsi",0))>75: risks.append("تشبع شرائي محتمل")
            if float(r.get("volatility",0))>60: risks.append("تقلب مرتفع")
        except Exception: pass
        if regime.get("label")=="Risk-Off": risks.append("السوق دفاعي")
    if not positives: positives=["مزيج متوازن من العوامل"]
    if not risks: risks=["لا توجد إشارة خطر كبيرة من البيانات المتاحة"]
    return "إيجابيات: "+"، ".join(positives[:4])+". مخاطر: "+"، ".join(risks[:3])+"."

def optional_llm_analysis(row,horizon,rule_text):
    key=os.getenv("OPENAI_API_KEY")
    model=os.getenv("OPENAI_MODEL","gpt-5.6-sol")
    if not key: return rule_text
    try:
        from openai import OpenAI
        client=OpenAI(api_key=key)
        keys=["Ticker","Company","Sector","LongScore","ShortScore","LongConfidence","ShortConfidence","technical","momentum",
              "growth","quality","valuation","sector_value_score","cashflow","balance","earnings_score","news_score",
              "risk","rsi","volatility","analyst_upside","atm_iv","put_call_oi"]
        compact={k:row.get(k) for k in keys}
        prompt=f"""حلل البيانات التالية لأغراض بحثية تعليمية فقط. لا تعط وعدًا ولا توصية شخصية ولا تدعي اليقين.
الأفق={horizon}
البيانات={json.dumps(compact,ensure_ascii=False,default=str)}
التحليل المحلي={rule_text}
أعطني: خلاصة، 3 نقاط قوة، 3 مخاطر، ما الذي يبطل الفرضية، ودرجة ثقة مع تفسيرها."""
        resp=client.responses.create(model=model,input=prompt)
        return resp.output_text.strip()
    except Exception:
        return rule_text
