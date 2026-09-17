# MT5 Mobile Bot Backend

This wraps the supplied `mt5_scalper_bot.py` behind a small FastAPI service. Run the trading engine and API on a Windows machine/VPS with MT5 installed and logged into a DEMO account.

## Run

1. Install Python 3.11+ and MetaTrader 5 desktop.
2. Put the supplied bot and this folder on the same machine.
3. `python -m venv .venv`
4. Windows: `.venv\\Scripts\\activate`
5. `pip install -r requirements.txt`
6. Copy `.env.example` to `.env` and set a strong API_TOKEN.
7. Keep `DRY_RUN=true` while testing.
8. Start API: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
9. Start the trading bot separately: `python mt5_scalper_bot.py`

For production, add HTTPS/reverse proxy, persistent database state, authentication/rotation, audit logs and a kill switch before live trading.
