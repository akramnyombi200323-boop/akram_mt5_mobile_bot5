import os, sqlite3, threading, time
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import MetaTrader5 as mt5

API_TOKEN = os.getenv("API_TOKEN", "change-me")
DB_PATH = os.getenv("DB_PATH", "bot_state.db")
app = FastAPI(title="Akram MT5 Mobile Bot API", version="2.0.0")
lock = threading.Lock()

class BotSettings(BaseModel):
    symbols: list[str] = ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD"]
    timeframe: int = Field(5, description="1,2,3,4,5,6,10,12,15,20,30")
    risk_per_trade: float = Field(0.0025, gt=0, le=0.05)
    max_daily_loss: float = Field(0.02, gt=0, le=0.20)
    max_open_positions: int = Field(3, ge=1, le=50)
    max_spread_points: float = Field(25, gt=0)
    cooldown_minutes: int = Field(10, ge=0)
    atr_sl_mult: float = Field(1.5, gt=0)
    atr_tp_mult: float = Field(2.0, gt=0)
    dry_run: bool = True

settings = BotSettings()
bot_running = False


def db():
    c = sqlite3.connect(DB_PATH)
    c.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts TEXT, kind TEXT, symbol TEXT, side TEXT, price REAL, sl REAL, tp REAL, volume REAL, profit REAL, message TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    c.commit(); return c


def auth(token: Optional[str]):
    if not API_TOKEN or token != API_TOKEN:
        raise HTTPException(401, "Invalid API token")


def mt5_info():
    try: return mt5.account_info()
    except Exception: return None


def mt5_positions():
    try: return mt5.positions_get() or ()
    except Exception: return ()


@app.get("/health")
def health():
    info = mt5.terminal_info()
    return {"ok": True, "mt5_initialized": bool(info), "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/status")
def status(x_api_token: Optional[str] = Header(default=None)):
    auth(x_api_token); info = mt5_info(); positions = mt5_positions()
    return {"running": bot_running, "dry_run": settings.dry_run, "balance": float(info.balance) if info else None,
            "equity": float(info.equity) if info else None, "profit": float(sum(getattr(p,"profit",0) for p in positions)),
            "open_positions": len(positions), "mt5_connected": bool(mt5.terminal_info()), "time": datetime.now(timezone.utc).isoformat()}


@app.post("/bot/start")
def start(x_api_token: Optional[str] = Header(default=None)):
    global bot_running
    auth(x_api_token); bot_running = True
    return {"running": True, "message": "Bot monitoring started"}


@app.post("/bot/stop")
def stop(x_api_token: Optional[str] = Header(default=None)):
    global bot_running
    auth(x_api_token); bot_running = False
    return {"running": False, "message": "Bot stopped; existing positions are not automatically closed"}


@app.post("/bot/emergency-stop")
def emergency_stop(x_api_token: Optional[str] = Header(default=None)):
    global bot_running
    auth(x_api_token); bot_running = False
    return {"running": False, "message": "Emergency stop activated. No new orders should be placed."}


@app.get("/positions")
def positions(x_api_token: Optional[str] = Header(default=None)):
    auth(x_api_token); rows=[]
    for p in mt5_positions():
        rows.append({"ticket": int(p.ticket), "symbol": p.symbol, "type": int(p.type), "volume": float(p.volume),
                     "price_open": float(p.price_open), "sl": float(p.sl), "tp": float(p.tp), "profit": float(p.profit)})
    return rows


@app.get("/settings")
def get_settings(x_api_token: Optional[str] = Header(default=None)):
    auth(x_api_token); return settings.model_dump()


@app.put("/settings")
def update_settings(body: BotSettings, x_api_token: Optional[str] = Header(default=None)):
    global settings
    auth(x_api_token)
    allowed={1,2,3,4,5,6,10,12,15,20,30}
    if body.timeframe not in allowed: raise HTTPException(400, "Unsupported timeframe")
    settings=body; return settings.model_dump()


@app.get("/signals")
def signals(x_api_token: Optional[str] = Header(default=None)):
    auth(x_api_token)
    # Reuse the supplied strategy module when available. Signals are informational in this V2 API.
    try:
        import mt5_scalper_bot as bot
        out=[]
        old_tf=bot.CFG.timeframe
        bot.CFG.timeframe=settings.timeframe
        for symbol in settings.symbols:
            df=bot.get_rates(symbol)
            if df is None: continue
            df=bot.add_indicators(df); x=df.iloc[-1]
            side=bot.signal(df)
            out.append({"symbol":symbol,"signal":side or "NONE","rsi":float(x.rsi),"atr":float(x.atr),
                        "ema_fast":float(x.ema_fast),"ema_slow":float(x.ema_slow),"ema_trend":float(x.ema_trend),
                        "close":float(x.close),"time":str(x.time)})
        bot.CFG.timeframe=old_tf
        return out
    except Exception as e:
        return {"error": str(e), "signals": []}


@app.get("/history")
def history(limit:int=50, x_api_token: Optional[str] = Header(default=None)):
    auth(x_api_token); c=db(); rows=c.execute("SELECT id,ts,kind,symbol,side,price,sl,tp,volume,profit,message FROM events ORDER BY id DESC LIMIT ?",(min(limit,200),)).fetchall(); c.close()
    return [dict(zip(["id","ts","kind","symbol","side","price","sl","tp","volume","profit","message"],r)) for r in rows]


@app.post("/paper-trade")
def paper_trade(body:dict, x_api_token: Optional[str] = Header(default=None)):
    auth(x_api_token)
    required=["symbol","side","price","sl","tp","volume"]
    if any(k not in body for k in required): raise HTTPException(400,"Missing trade field")
    c=db(); c.execute("INSERT INTO events(ts,kind,symbol,side,price,sl,tp,volume,profit,message) VALUES(?,?,?,?,?,?,?,?,?,?)",
                      (datetime.now(timezone.utc).isoformat(),"PAPER",body["symbol"],body["side"],body["price"],body["sl"],body["tp"],body["volume"],0,"Demo/paper trade")); c.commit(); c.close()
    return {"ok":True}
