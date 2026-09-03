# Production Checklist

Before real-money use:

- [ ] Run `python tests.py`
- [ ] Run `python healthcheck.py`
- [ ] Run paper-trading only for at least several weeks
- [ ] Confirm daily data health > 75
- [ ] Confirm Top 5 lists have 5 valid diversified candidates
- [ ] Track 1D/5D/21D/63D alpha vs benchmark
- [ ] Reject model if persistent negative alpha emerges
- [ ] Review provider-schema errors in `market_intelligence.log`
- [ ] Never commit `.env`
- [ ] Use environment/secrets for API keys
- [ ] Use persistent DB in production hosting
- [ ] Add provider SLA/paid feed before institutional use
- [ ] Confirm earnings blackout and options-data availability
- [ ] Validate position sizing independently before any order
- [ ] No automatic brokerage execution is included by design

A system can be production-ready in engineering terms without being guaranteed profitable.
