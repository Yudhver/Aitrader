"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║   NIFTY 50 AI SWING TRADING AGENT  v4.4                                        ║
║   Broker      : SmartAPI (Angel One)                                            ║
║   AI Engines  : Gemini 2.0 Flash (google-genai SDK) + Groq Qwen3-32B           ║
║   Framework   : John J. Murphy — Technical Analysis of Financial Markets        ║
║   Strategy    : Mean Reversion + Breakout Retest + Golden Cross                 ║
║                 Multi-Timeframe Confluence (Daily + Hourly + 15-Min)            ║
║   Scoring     : Dynamic threshold (market-regime adaptive)                      ║
║   Filters     : Volume ≥1.2× avg | FII net gate | Relative strength             ║
║                 India VIX gate | Weekly trend alignment                          ║
║   Sizing      : Risk 2% of current capital | ATR-based Dynamic SL               ║
║   AI Roles    : Both AIs score fundamentals + technicals independently          ║
║                 Consensus required for AI bonus (+2 bull / -3 bear)             ║
║   Threading   : Single-threaded data fetch | Parallel indicator computation     ║
║   Execution   : Paper Trading Mode — Persistent Virtual Ledger                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝

REQUIRED PACKAGES:
    pip install smartapi-python pyotp pandas pandas_ta google-genai
    pip install rich requests python-dotenv logzero numpy groq

.env FILE (~/Desktop/.env):
    ANGEL_API_KEY=your_api_key
    ANGEL_CLIENT_CODE=your_client_code
    ANGEL_PASSWORD=your_mpin
    ANGEL_TOTP_SECRET=your_totp_secret
    GEMINI_API_KEY=your_gemini_key
    GROQ_API_KEY=your_groq_key          ← free at console.groq.com

CHANGES FROM v4.3 (MASTER FILE FOR INDEX TOKENS):
    ✓ Fix P — fetch_token_map() now extracts sector index tokens from the same
               OpenAPIScripMaster.json download in a single pass (instrumenttype
               == "AMXIDX", exch_seg == "NSE"), so tokens are always live and
               never stale. No second HTTP request needed.
    ✓ Fix Q — sector_index_token_map global populated at startup alongside
               token_map; get_sector_scores() uses it instead of the hardcoded
               SECTOR_INDICES dict. SECTOR_INDICES now serves only as a fallback
               if the master file doesn't contain a particular index.
    ✓ Fix R — fetch_token_map() reports both equity and index mapping counts
               separately so mismatches are immediately visible in console output.

CHANGES FROM v4.2 (SECTOR INDEX TOKEN FIX):
    ✓ Fix O — All 8 sector index tokens updated from deprecated 26xxx series to
               new 99926xxx series. Old tokens return status=False, message="SUCCESS"
               which caused all sector fetches to silently fail after 3 retries.
               Updated mapping:
               NIFTY IT     26000  → 99926008
               NIFTY BANK   26009  → 99926009
               NIFTY PHARMA 26013  → 99926023
               NIFTY ENERGY 26015  → 99926020
               NIFTY FMCG   26035  → 99926021
               NIFTY AUTO   26037  → 99926029
               NIFTY METAL  26042  → 99926030
               NIFTY REALTY 26049  → 99926018

CHANGES FROM v4.1 (GEMINI SDK MIGRATION):
    ✓ Fix J — Migrated from deprecated google-generativeai (EOL Nov 30 2025) to
               the new unified google-genai SDK (pip install google-genai)
    ✓ Fix K — genai.configure() + GenerativeModel() replaced with:
               google_genai.Client(api_key=...) + client.models.generate_content()
    ✓ Fix L — genai.GenerationConfig() replaced with genai_types.GenerateContentConfig()
    ✓ Fix M — _gemini_model global replaced with _gemini_client (Client instance)
    ✓ Fix N — Import changed: import google.generativeai as genai →
               from google import genai as google_genai + from google.genai import types

CHANGES FROM v4.0 (DATA FETCH FIX PATCH):
    BUG FIXES:
    ✓ Fix A — init_broker() now returns Optional[SmartConnect] (was returning None always)
    ✓ Fix B — main() declares global smartApi, captures return value, exits on failure
    ✓ Fix C — _fetch_candles() guards against smartApi=None before any API call
    ✓ Fix D — get_nifty_daily_data() guards against smartApi=None
    ✓ Fix E — ONE_WEEK interval removed (SmartAPI does not support it); weekly candles
               now derived by resampling daily data with pandas resample("W-FRI")
    ✓ Fix F — Groq model updated from deprecated qwen-qwq-32b → qwen/qwen3-32b
               with reasoning_format="hidden" for clean JSON output (no <think> leakage)
    ✓ Fix G — Timestamp parsing uses UTC-aware conversion (tz_convert → tz_localize)
               to avoid pandas warnings and ensure correct IST timestamps
    ✓ Fix H — SmartAPI status check extended to handle both True and "true" strings
               consistently across all getCandleData call sites
    ✓ Fix I — _fetch_candles logs errorcode and message from failed responses for
               easier debugging of AB1004/AB13000 API errors

CHANGES FROM v3.0 (carried forward):
    BUG FIXES:
    ✓ Fix 1 — Swing high validated against current price (negative R:R eliminated)
    ✓ Fix 2 — analysis dict copied before mutation in enter_trade()
    ✓ Fix 3 — compute_indicators called once, result passed down (no double-compute)
    ✓ Fix 4 — 15-min score now contributes +1 (not += 0 dead code)
    ✓ Fix 5 — Position sizing uses live ledger capital, not constant PAPER_CAPITAL
    ✓ Fix 6 — Bollinger Band columns use BBL_20_2.0 (pandas_ta actual names)

    AI ARCHITECTURE:
    ✓ Dual AI consensus: Gemini 2.0 Flash + Groq Qwen3-32B
    ✓ Both AIs see identical context: price, indicators, sector, market regime
    ✓ Both return identical JSON: fundamental_bias, technical_bias, overall_bias,
      suggested_sl, suggested_tp1, suggested_tp2, confidence, earnings_risk,
      earnings_note, sector_outlook, reasoning
    ✓ AI-suggested SL with safety rules (1%–4% below entry, not above formula SL)
    ✓ Two-target TP: tp1 (partial exit 50%) + tp2 (full exit), capped at 15%

    NEW FILTERS (HIGH IMPACT):
    ✓ Volume confirmation: entry only if volume ≥ 1.2× 20-day average
    ✓ FII/DII net flow gate: skip new entries on days FII net sell > ₹500 Cr
    ✓ Relative strength: stock must outperform Nifty over last 5 days

    NEW FILTERS (MEDIUM IMPACT):
    ✓ Weekly trend alignment: daily trade must match weekly trend direction
    ✓ India VIX gate: no new entries if VIX > 20
    ✓ Dynamic score threshold based on Nifty market regime

    OTHER IMPROVEMENTS:
    ✓ Sector rotation: 8 sector indices fetched, RSI computed, passed to AI
    ✓ Trailing stop loss: activates after 1.5× ATR gain, trails to breakeven+
    ✓ Portfolio heat: no two stocks from same sector simultaneously (max 2)
    ✓ ThreadPoolExecutor for parallel indicator computation (CPU-bound)
    ✓ All data fetching remains single-threaded (SmartAPI rate limit safe)
"""

# ─────────────────────────────────────────────────────────────────────────────
#  STANDARD LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import time
import logging
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
#  THIRD-PARTY LIBRARIES
# ─────────────────────────────────────────────────────────────────────────────
try:
    import requests
    import pyotp
    import pandas as pd
    import pandas_ta as ta
    import numpy as np
    from dotenv import load_dotenv
    from SmartApi import SmartConnect
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.align import Align
    from rich import box
    from rich.text import Text
except ImportError as e:
    print(f"\n[ERROR] Missing library: {e}")
    print("Run: pip install smartapi-python pyotp pandas pandas_ta google-genai "
          "rich requests python-dotenv numpy groq")
    sys.exit(1)

# Optional Gemini — uses new unified google-genai SDK (google-generativeai is EOL Nov 2025)
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Optional Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONSOLE & LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
console = Console()
OUTPUT_DIR = Path.home() / "Desktop" / "NiftyAI"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NiftyAI")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
PAPER_CAPITAL        = 500_000.0   # ₹5 Lakhs starting capital
RISK_PER_TRADE       = 0.02        # 2% risk per trade
MAX_OPEN_POSITIONS   = 5           # Max concurrent paper positions
MAX_SECTOR_EXPOSURE  = 2           # Max positions per sector
MAX_RETRIES          = 3           # API call retries
RATE_LIMIT_DELAY     = 0.6         # Seconds between SmartAPI calls
DAILY_LOOKBACK       = 260         # ~1 year of daily data
HOURLY_LOOKBACK      = 30          # Days for hourly data
WEEKLY_LOOKBACK      = 60          # Days for weekly data (~12 weeks)
MIN_CANDLES          = 210         # Min candles for 200-SMA
CALC_WORKERS         = 4           # Threads for parallel indicator computation

# Dynamic threshold — overridden at runtime by get_market_regime()
SCORE_THRESHOLD      = 4

# AI safety rules for SL/TP
AI_SL_MAX_PCT        = 0.04        # AI SL cannot be more than 4% below entry
AI_SL_MIN_PCT        = 0.01        # AI SL cannot be less than 1% below entry
AI_TP_MAX_PCT        = 0.15        # AI TP cannot be more than 15% above entry

# FII gate threshold (crores) — skip new entries when FII net sell > this
FII_SELL_THRESHOLD_CR = 500.0

# Volume confirmation
VOLUME_CONFIRM_RATIO  = 1.2        # Entry only if volume ≥ 1.2× 20-day avg

# India VIX gate
VIX_MAX_FOR_ENTRY     = 20.0       # No entries if VIX > 20

# File paths
STATE_FILE   = OUTPUT_DIR / "paper_ledger.json"
TRADES_CSV   = OUTPUT_DIR / "trade_log.csv"
SUMMARY_FILE = OUTPUT_DIR / "daily_summary.json"

# ─────────────────────────────────────────────────────────────────────────────
#  NIFTY 50 SYMBOLS
# ─────────────────────────────────────────────────────────────────────────────
NIFTY_50_SYMBOLS = [
    "RELIANCE-EQ",   "HDFCBANK-EQ",   "INFY-EQ",       "TCS-EQ",        "ICICIBANK-EQ",
    "HINDUNILVR-EQ", "SBIN-EQ",       "BHARTIARTL-EQ", "KOTAKBANK-EQ",  "BAJFINANCE-EQ",
    "LT-EQ",         "ASIANPAINT-EQ", "AXISBANK-EQ",   "MARUTI-EQ",     "TITAN-EQ",
    "SUNPHARMA-EQ",  "ULTRACEMCO-EQ", "TECHM-EQ",      "WIPRO-EQ",      "NESTLEIND-EQ",
    "POWERGRID-EQ",  "NTPC-EQ",       "ONGC-EQ",       "COALINDIA-EQ",  "ADANIENT-EQ",
    "ADANIPORTS-EQ", "BAJAJFINSV-EQ", "BRITANNIA-EQ",  "CIPLA-EQ",      "DIVISLAB-EQ",
    "DRREDDY-EQ",    "EICHERMOT-EQ",  "GRASIM-EQ",     "HCLTECH-EQ",    "HEROMOTOCO-EQ",
    "HINDALCO-EQ",   "INDUSINDBK-EQ", "JSWSTEEL-EQ",   "M&M-EQ",        "SBILIFE-EQ",
    "TATACONSUM-EQ", "TATAMOTORS-EQ", "TATASTEEL-EQ",  "BPCL-EQ",       "IOC-EQ",
    "APOLLOHOSP-EQ", "BAJAJ-AUTO-EQ", "HDFCLIFE-EQ",   "ITC-EQ",        "SHRIRAMFIN-EQ",
]

# Sector map for portfolio heat check
SYMBOL_SECTOR = {
    "RELIANCE-EQ": "Energy",      "ONGC-EQ": "Energy",         "BPCL-EQ": "Energy",
    "IOC-EQ": "Energy",           "COALINDIA-EQ": "Energy",     "NTPC-EQ": "Energy",
    "POWERGRID-EQ": "Utilities",  "ADANIENT-EQ": "Conglomerate",
    "HDFCBANK-EQ": "Banking",     "ICICIBANK-EQ": "Banking",    "SBIN-EQ": "Banking",
    "KOTAKBANK-EQ": "Banking",    "AXISBANK-EQ": "Banking",     "INDUSINDBK-EQ": "Banking",
    "BAJFINANCE-EQ": "Finance",   "BAJAJFINSV-EQ": "Finance",   "SBILIFE-EQ": "Finance",
    "HDFCLIFE-EQ": "Finance",     "SHRIRAMFIN-EQ": "Finance",
    "TCS-EQ": "IT",               "INFY-EQ": "IT",              "WIPRO-EQ": "IT",
    "HCLTECH-EQ": "IT",           "TECHM-EQ": "IT",
    "SUNPHARMA-EQ": "Pharma",     "CIPLA-EQ": "Pharma",         "DIVISLAB-EQ": "Pharma",
    "DRREDDY-EQ": "Pharma",       "APOLLOHOSP-EQ": "Healthcare",
    "HINDUNILVR-EQ": "FMCG",      "NESTLEIND-EQ": "FMCG",       "BRITANNIA-EQ": "FMCG",
    "TATACONSUM-EQ": "FMCG",      "ITC-EQ": "FMCG",
    "MARUTI-EQ": "Auto",          "TATAMOTORS-EQ": "Auto",      "HEROMOTOCO-EQ": "Auto",
    "EICHERMOT-EQ": "Auto",       "BAJAJ-AUTO-EQ": "Auto",      "M&M-EQ": "Auto",
    "TATASTEEL-EQ": "Metal",      "JSWSTEEL-EQ": "Metal",       "HINDALCO-EQ": "Metal",
    "GRASIM-EQ": "Cement",        "ULTRACEMCO-EQ": "Cement",
    "LT-EQ": "Infra",             "ADANIPORTS-EQ": "Infra",
    "ASIANPAINT-EQ": "Consumer",  "TITAN-EQ": "Consumer",
    "BHARTIARTL-EQ": "Telecom",
}

# Sector index names → fallback hardcoded tokens (99926xxx series).
# fetch_token_map() will overwrite these with live tokens from the master file
# (instrumenttype == "AMXIDX", exch_seg == "NSE") so this dict is only used
# if the master file lookup fails for a particular index.
SECTOR_INDICES = {
    "NIFTY IT":     "99926008",
    "NIFTY BANK":   "99926009",
    "NIFTY PHARMA": "99926023",
    "NIFTY ENERGY": "99926020",
    "NIFTY FMCG":   "99926021",
    "NIFTY AUTO":   "99926029",
    "NIFTY METAL":  "99926030",
    "NIFTY REALTY": "99926018",
}

# Live sector index token map — populated by fetch_token_map() from master file.
# Maps the same keys as SECTOR_INDICES → token string.
sector_index_token_map: dict = {}

# ─────────────────────────────────────────────────────────────────────────────
#  BROKER AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path.home() / "Desktop" / ".env")

smartApi    : Optional[SmartConnect] = None
token_map   : dict = {}
jwt_token   : str  = ""
refresh_tok : str  = ""
feed_token  : str  = ""


def init_broker() -> Optional[SmartConnect]:
    """
    Login to SmartAPI with TOTP and store session tokens.
    Returns SmartConnect object on success, None on failure.
    """
    global jwt_token, refresh_tok, feed_token

    api_key     = os.getenv("ANGEL_API_KEY")
    client_code = os.getenv("ANGEL_CLIENT_CODE")
    password    = os.getenv("ANGEL_PASSWORD")
    totp_secret = os.getenv("ANGEL_TOTP_SECRET")

    if not all([api_key, client_code, password, totp_secret]):
        console.print(Panel(
            "[bold red]Missing credentials in .env\n"
            "Required: ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_PASSWORD, ANGEL_TOTP_SECRET[/]",
            title="[red]AUTH ERROR[/]", border_style="red"
        ))
        return None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            api     = SmartConnect(api_key=api_key)
            totp    = pyotp.TOTP(totp_secret).now()
            session = api.generateSession(client_code, password, totp)

            if session.get("status") is True or session.get("status") == "true":
                jwt_token   = session["data"]["jwtToken"]
                refresh_tok = session["data"]["refreshToken"]
                feed_token  = api.getfeedToken()
                api.generateToken(refresh_tok)

                name = session["data"].get("name", client_code)
                console.print(Panel(
                    f"[bold green]✓ Connected as: {name}[/]\n"
                    f"[dim]JWT: {jwt_token[:30]}...[/]",
                    title="[green]SmartAPI LOGIN[/]", border_style="green"
                ))
                return api   # ← Return the live API object
            else:
                msg = session.get("message", "Unknown error")
                raise ValueError(f"Login returned status=False: {msg}")

        except Exception as e:
            wait = 2 ** attempt
            console.print(f"[red]  [Attempt {attempt}/{MAX_RETRIES}] Login failed: {e} — retrying in {wait}s[/]")
            if attempt == MAX_RETRIES:
                console.print("[bold red]All login attempts exhausted. Check credentials.[/]")
                return None
            time.sleep(wait)

    return None


def fetch_token_map() -> None:
    """
    Download Angel One instrument master (single HTTP call) and populate:
      • token_map            — Nifty 50 equity symbols → token
      • sector_index_token_map — sector index names → token  (AMXIDX instruments)

    Both maps are built in one pass over the master JSON, so we never download
    the file twice. Hardcoded fallback tokens in SECTOR_INDICES are only used
    for any index the master file doesn't contain.
    """
    global token_map, sector_index_token_map
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

    # Build a quick lookup: index display-name → SECTOR_INDICES key
    # Master file uses names like "Nifty IT", "Nifty Bank" etc.
    # We match case-insensitively against our SECTOR_INDICES keys.
    index_name_lookup = {k.lower(): k for k in SECTOR_INDICES}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            console.print("[dim]  Fetching instrument master from Angel One...[/]")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            master = resp.json()

            eq_found  = 0
            idx_found = 0

            for item in master:
                sym      = item.get("symbol", "")
                seg      = item.get("exch_seg", "")
                itype    = item.get("instrumenttype", "")
                name     = item.get("name", "")
                token    = item.get("token", "")

                # ── Nifty 50 equities ────────────────────────────────────────
                if seg == "NSE" and sym in NIFTY_50_SYMBOLS:
                    token_map[sym] = token
                    eq_found += 1

                # ── Sector indices (AMXIDX type, NSE segment) ────────────────
                if seg == "NSE" and itype == "AMXIDX" and token:
                    key = index_name_lookup.get(name.strip().lower())
                    if key:
                        sector_index_token_map[key] = token
                        idx_found += 1

            # Apply fallbacks for any index not found in master
            for idx_key, fallback_token in SECTOR_INDICES.items():
                if idx_key not in sector_index_token_map:
                    sector_index_token_map[idx_key] = fallback_token
                    logger.warning(
                        f"Index '{idx_key}' not found in master file — "
                        f"using fallback token {fallback_token}"
                    )

            console.print(
                f"[green]  ✓ Mapped {eq_found}/{len(NIFTY_50_SYMBOLS)} Nifty 50 equity tokens[/]"
            )
            console.print(
                f"[green]  ✓ Mapped {idx_found}/{len(SECTOR_INDICES)} sector index tokens "
                f"from master file[/]"
            )
            if eq_found < len(NIFTY_50_SYMBOLS):
                missing = [s for s in NIFTY_50_SYMBOLS if s not in token_map]
                console.print(f"[yellow]  ⚠ Unmapped equities (check if delisted): {missing}[/]")
            return

        except Exception as e:
            wait = 2 ** attempt
            console.print(f"[red]  [Attempt {attempt}/{MAX_RETRIES}] Token fetch failed: {e}[/]")
            if attempt == MAX_RETRIES:
                console.print("[bold red]Failed to fetch token map. Cannot proceed.[/]")
                sys.exit(1)
            time.sleep(wait)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA FETCHING  (single-threaded — SmartAPI rate limit compliance)
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_candles(symbol: str, interval: str, from_dt: datetime,
                   to_dt: datetime, token_override: str = None) -> pd.DataFrame:
    """
    Core data fetcher with retries + exponential backoff.
    Single-threaded: all calls sleep RATE_LIMIT_DELAY before firing.
    token_override: used for sector index tokens not in token_map.

    SmartAPI max day ranges per interval:
      ONE_MINUTE=30, THREE/FIVE/TEN_MINUTE=90, FIFTEEN/THIRTY_MINUTE=180,
      ONE_HOUR=365, ONE_DAY=2000
    NOTE: ONE_WEEK is NOT a supported interval — use resample on daily data.
    """
    # Guard: smartApi must be initialised
    if smartApi is None:
        logger.error(f"[{symbol}] SmartAPI not initialised — cannot fetch data")
        return pd.DataFrame()

    token = token_override or token_map.get(symbol)
    if not token:
        logger.warning(f"[{symbol}] Token not found in map — skipping")
        return pd.DataFrame()

    params = {
        "exchange":    "NSE",
        "symboltoken": token,
        "interval":    interval,
        "fromdate":    from_dt.strftime("%Y-%m-%d %H:%M"),
        "todate":      to_dt.strftime("%Y-%m-%d %H:%M"),
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            resp = smartApi.getCandleData(params)

            if resp and (resp.get("status") is True or resp.get("status") == "true") and resp.get("data"):
                df = pd.DataFrame(
                    resp["data"],
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
                df = df.set_index("timestamp").sort_index()
                df = df.apply(pd.to_numeric, errors="coerce").dropna()
                return df

            error_msg = resp.get("message", "Unknown error") if resp else "None response"
            error_code = resp.get("errorcode", "") if resp else ""
            logger.warning(f"[{symbol}] {interval} bad response [{error_code}]: {error_msg}")
            raise ValueError(f"Bad response [{error_code}]: {error_msg}")

        except Exception as e:
            if attempt == MAX_RETRIES:
                logger.error(f"[{symbol}] {interval} fetch failed after {MAX_RETRIES} retries: {e}")
                return pd.DataFrame()
            wait = 2 ** attempt
            logger.debug(f"[{symbol}] {interval} retry {attempt}/{MAX_RETRIES} in {wait}s: {e}")
            time.sleep(wait)

    return pd.DataFrame()


def get_daily_data(symbol: str) -> pd.DataFrame:
    to_dt   = datetime.now() - timedelta(minutes=5)
    from_dt = to_dt - timedelta(days=int(DAILY_LOOKBACK * 1.5))
    return _fetch_candles(symbol, "ONE_DAY", from_dt, to_dt)


def get_weekly_data(symbol: str) -> pd.DataFrame:
    """
    Derive weekly candles by resampling daily data.
    SmartAPI does NOT support a ONE_WEEK interval — we build it ourselves.
    Weekly candles: open=first, high=max, low=min, close=last, volume=sum.
    Uses the same daily data already fetched (or fetches fresh if needed).
    """
    to_dt   = datetime.now() - timedelta(minutes=5)
    from_dt = to_dt - timedelta(days=int(WEEKLY_LOOKBACK * 7 * 1.2))  # extra buffer
    df = _fetch_candles(symbol, "ONE_DAY", from_dt, to_dt)
    if df.empty:
        return pd.DataFrame()
    weekly = df.resample("W-FRI").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return weekly


def get_hourly_data(symbol: str) -> pd.DataFrame:
    to_dt   = datetime.now() - timedelta(minutes=5)
    from_dt = to_dt - timedelta(days=HOURLY_LOOKBACK)
    return _fetch_candles(symbol, "ONE_HOUR", from_dt, to_dt)


def get_15min_data(symbol: str) -> pd.DataFrame:
    to_dt   = datetime.now() - timedelta(minutes=5)
    from_dt = to_dt - timedelta(days=10)
    return _fetch_candles(symbol, "FIFTEEN_MINUTE", from_dt, to_dt)


def get_sector_index_data(sector_name: str, token: str) -> pd.DataFrame:
    """Fetch 30 days of daily data for a sector index using its token directly."""
    to_dt   = datetime.now() - timedelta(minutes=5)
    from_dt = to_dt - timedelta(days=30)
    return _fetch_candles(sector_name, "ONE_DAY", from_dt, to_dt, token_override=token)


# ─────────────────────────────────────────────────────────────────────────────
#  TECHNICAL ANALYSIS ENGINE
#  NOTE: compute_indicators is called from parallel threads — pandas ops are
#  thread-safe for separate DataFrames (no shared mutable state).
# ─────────────────────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators on a copy of df.
    Returns enriched DataFrame. Always operates on a copy — thread-safe.
    FIX #3: Was called twice in v3. Now called once; result passed down.
    FIX #6: Use correct pandas_ta column names (BBL_20_2.0 not BBL_20).
    """
    if df.empty or len(df) < 20:
        return df.copy()

    df = df.copy()  # Never mutate caller's DataFrame

    df.ta.sma(length=50,  append=True)    # SMA_50
    df.ta.sma(length=200, append=True)    # SMA_200
    df.ta.ema(length=20,  append=True)    # EMA_20
    df.ta.ema(length=9,   append=True)    # EMA_9
    df.ta.rsi(length=14,  append=True)    # RSI_14
    df.ta.macd(append=True)               # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    df.ta.atr(length=14,  append=True)    # ATRr_14
    df.ta.bbands(length=20, append=True)  # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0

    # Volume ratio vs 20-day average
    df["vol_avg_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"]  = df["volume"] / df["vol_avg_20"].replace(0, np.nan)

    return df


def detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    """Detect Murphy-approved entry patterns on last 3 candles."""
    if len(df) < 3:
        return {}

    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    prev2 = df.iloc[-3]

    patterns = {}

    # Bullish Engulfing
    patterns["bullish_engulfing"] = (
        prev["close"] < prev["open"]
        and last["close"] > last["open"]
        and last["open"] < prev["close"]
        and last["close"] > prev["open"]
    )

    # Hammer
    body       = abs(last["close"] - last["open"])
    range_     = last["high"] - last["low"]
    lower_wick = min(last["open"], last["close"]) - last["low"]
    upper_wick = last["high"] - max(last["open"], last["close"])
    if range_ > 0:
        patterns["hammer"] = (
            lower_wick >= 2 * body
            and upper_wick <= 0.3 * body
            and last["close"] > last["open"]
        )
    else:
        patterns["hammer"] = False

    # Bearish Engulfing
    patterns["bearish_engulfing"] = (
        prev["close"] > prev["open"]
        and last["close"] < last["open"]
        and last["open"] > prev["close"]
        and last["close"] < prev["open"]
    )

    # Inside Bar
    patterns["inside_bar"] = (
        last["high"] < prev["high"]
        and last["low"] > prev["low"]
    )

    # Three White Soldiers
    patterns["three_white_soldiers"] = (
        prev2["close"] > prev2["open"]
        and prev["close"] > prev["open"]
        and last["close"] > last["open"]
        and prev["close"] > prev2["close"]
        and last["close"] > prev["close"]
    )

    return patterns


def find_swing_levels(df: pd.DataFrame, window: int = 10) -> dict:
    """Identify recent swing high/low using local maxima/minima."""
    if len(df) < window * 2:
        return {"swing_high": None, "swing_low": None}

    recent     = df.tail(60)
    highs      = recent["high"].rolling(window, center=True).max()
    lows       = recent["low"].rolling(window, center=True).min()
    swing_high = recent["high"][recent["high"] == highs].max()
    swing_low  = recent["low"][recent["low"] == lows].min()

    return {
        "swing_high": float(swing_high) if pd.notna(swing_high) else None,
        "swing_low":  float(swing_low)  if pd.notna(swing_low)  else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PARALLEL SCORING FUNCTIONS
#  These are pure computation (no I/O, no API calls) — safe to run in threads.
# ─────────────────────────────────────────────────────────────────────────────
def score_daily(df_enriched: pd.DataFrame) -> tuple[int, list[str]]:
    """
    Score daily timeframe. Receives pre-computed (enriched) DataFrame.
    FIX #3: No longer calls compute_indicators internally.
    """
    if df_enriched.empty or len(df_enriched) < MIN_CANDLES:
        return 0, ["Insufficient data"]

    lat = df_enriched.iloc[-1]
    prv = df_enriched.iloc[-2]

    score   = 0
    signals = []

    # TREND — Dow Theory
    if pd.notna(lat.get("SMA_200")) and lat["close"] > lat["SMA_200"]:
        score += 1
        signals.append("Above 200-SMA ↑")

    # Golden Cross
    if all(pd.notna(lat.get(c)) and pd.notna(prv.get(c))
           for c in ["SMA_50", "SMA_200"]):
        if lat["SMA_50"] > lat["SMA_200"] and prv["SMA_50"] <= prv["SMA_200"]:
            score += 2
            signals.append("🥇 Golden Cross!")

    # MOMENTUM — RSI
    rsi_now  = lat.get("RSI_14")
    rsi_prev = prv.get("RSI_14")
    if pd.notna(rsi_now) and pd.notna(rsi_prev):
        if 50 <= rsi_now <= 70 and rsi_now > rsi_prev:
            score += 1
            signals.append(f"RSI rising {rsi_prev:.1f}→{rsi_now:.1f}")
        elif rsi_now > 70:
            score -= 1
            signals.append(f"RSI overbought {rsi_now:.1f}")

    # MACD histogram crossover
    macd_h_now  = lat.get("MACDh_12_26_9")
    macd_h_prev = prv.get("MACDh_12_26_9")
    if pd.notna(macd_h_now) and pd.notna(macd_h_prev):
        if macd_h_now > 0 and macd_h_prev <= 0:
            score += 1
            signals.append("MACD Histogram crossover ↑")

    # MEAN REVERSION — EMA bounce
    ema20 = lat.get("EMA_20")
    if pd.notna(ema20) and pd.notna(rsi_now):
        if lat["low"] <= ema20 * 1.005 and lat["close"] > ema20 and 40 <= rsi_now <= 55:
            score += 2
            signals.append("20-EMA Bounce (Mean Rev)")

    # Candlestick at EMA
    patterns = detect_candlestick_patterns(df_enriched)
    if patterns.get("bullish_engulfing") or patterns.get("hammer"):
        if pd.notna(ema20) and lat["low"] <= ema20 * 1.01:
            score += 1
            signals.append("Candle: Engulfing/Hammer at EMA")
    if patterns.get("bearish_engulfing"):
        score -= 1
        signals.append("⚠ Bearish Engulfing (negative)")

    # BREAKOUT RETEST — 50-SMA
    sma50 = lat.get("SMA_50")
    if pd.notna(sma50):
        if prv["close"] < sma50 and lat["close"] > sma50:
            vol_r = lat.get("vol_ratio", 1.0)
            if pd.notna(vol_r) and vol_r >= 1.3:
                score += 2
                signals.append("50-SMA Breakout (Vol confirmed)")
            else:
                score += 1
                signals.append("50-SMA Breakout (low vol)")

    # Three White Soldiers
    if patterns.get("three_white_soldiers"):
        score += 1
        signals.append("Three White Soldiers ↑")

    return score, signals


def score_hourly(df: pd.DataFrame) -> tuple[int, list[str]]:
    """Hourly timeframe — computed in parallel thread."""
    if df.empty or len(df) < 50:
        return 0, []

    df_e = compute_indicators(df)
    lat  = df_e.iloc[-1]

    score   = 0
    signals = []

    ema20 = lat.get("EMA_20")
    sma50 = lat.get("SMA_50")
    rsi   = lat.get("RSI_14")

    if pd.notna(sma50) and lat["close"] > sma50:
        score += 1
        signals.append("H1: Above 50-SMA")

    if pd.notna(rsi) and 45 <= rsi <= 65:
        score += 1
        signals.append(f"H1: RSI healthy {rsi:.1f}")

    ema9 = lat.get("EMA_9")
    if pd.notna(ema9) and pd.notna(ema20) and ema9 > ema20:
        score += 1
        signals.append("H1: EMA9>EMA20 (momentum)")

    return score, signals


def score_15min(df: pd.DataFrame) -> tuple[int, list[str]]:
    """
    15-min entry timing — computed in parallel thread.
    FIX #4: Now contributes +1 (was += 0 dead code in v3).
    """
    if df.empty or len(df) < 20:
        return 0, []

    df_e = compute_indicators(df)
    lat  = df_e.iloc[-1]

    score   = 0
    signals = []

    rsi  = lat.get("RSI_14")
    macd = lat.get("MACD_12_26_9")
    msig = lat.get("MACDs_12_26_9")

    if pd.notna(rsi) and 40 <= rsi <= 65:
        score += 1
        signals.append("15m: RSI clear entry zone")

    if pd.notna(macd) and pd.notna(msig) and macd > msig:
        score += 1
        signals.append("15m: MACD bullish")

    return score, signals


def check_volume_confirmation(df_enriched: pd.DataFrame) -> tuple[bool, float]:
    """
    HIGH IMPACT FILTER: Entry only if last candle volume ≥ 1.2× 20-day average.
    Returns (passes: bool, ratio: float).
    """
    if df_enriched.empty:
        return True, 1.0  # Neutral if no data

    lat   = df_enriched.iloc[-1]
    ratio = float(lat.get("vol_ratio", 1.0) or 1.0)
    return ratio >= VOLUME_CONFIRM_RATIO, ratio


def check_relative_strength(df_stock: pd.DataFrame, df_nifty: pd.DataFrame,
                             lookback: int = 5) -> tuple[bool, float]:
    """
    HIGH IMPACT FILTER: Stock must outperform Nifty over last `lookback` days.
    Returns (outperforms: bool, rs_score: float).
    RS score > 1.0 means stock beat Nifty.
    """
    if df_stock.empty or df_nifty.empty:
        return True, 1.0  # Neutral if data missing

    try:
        stock_ret = df_stock["close"].iloc[-1] / df_stock["close"].iloc[-(lookback+1)] - 1
        nifty_ret = df_nifty["close"].iloc[-1] / df_nifty["close"].iloc[-(lookback+1)] - 1
        rs = (1 + stock_ret) / (1 + nifty_ret) if (1 + nifty_ret) != 0 else 1.0
        return rs > 1.0, round(rs, 4)
    except (IndexError, ZeroDivisionError):
        return True, 1.0


def check_weekly_trend(df_weekly: pd.DataFrame) -> tuple[bool, str]:
    """
    MEDIUM IMPACT FILTER: Daily bullish trade only if weekly trend is also bullish.
    Returns (aligned: bool, weekly_bias: str).
    Weekly bullish = price above weekly 20-EMA AND weekly RSI > 50.
    """
    if df_weekly.empty or len(df_weekly) < 15:
        return True, "UNKNOWN"  # Neutral if insufficient data

    try:
        df_w = compute_indicators(df_weekly)
        lat  = df_w.iloc[-1]

        ema20 = lat.get("EMA_20")
        rsi   = lat.get("RSI_14")

        if pd.notna(ema20) and pd.notna(rsi):
            bullish = lat["close"] > ema20 and rsi > 50
            bias    = "BULLISH" if bullish else "BEARISH"
            return bullish, bias
    except Exception as e:
        logger.debug(f"Weekly trend check failed: {e}")

    return True, "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
#  MARKET-WIDE DATA (single-threaded, fetched once per run)
# ─────────────────────────────────────────────────────────────────────────────
_nifty_data_cache: Optional[pd.DataFrame] = None
_market_regime_cache: Optional[dict] = None
_sector_scores_cache: Optional[dict] = None
_india_vix_cache: Optional[float] = None
_fii_net_cache: Optional[float] = None


def get_nifty_daily_data() -> pd.DataFrame:
    """Fetch Nifty 50 index daily data. Cached for the run."""
    global _nifty_data_cache
    if _nifty_data_cache is not None:
        return _nifty_data_cache

    # Nifty 50 index token in SmartAPI
    to_dt   = datetime.now() - timedelta(minutes=5)
    from_dt = to_dt - timedelta(days=int(DAILY_LOOKBACK * 1.5))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if smartApi is None:
                logger.error("SmartAPI not initialised — cannot fetch Nifty index data")
                return pd.DataFrame()

            time.sleep(RATE_LIMIT_DELAY)
            params = {
                "exchange":    "NSE",
                "symboltoken": "99926000",   # Nifty 50 index token
                "interval":    "ONE_DAY",
                "fromdate":    from_dt.strftime("%Y-%m-%d %H:%M"),
                "todate":      to_dt.strftime("%Y-%m-%d %H:%M"),
            }
            resp = smartApi.getCandleData(params)
            if resp and (resp.get("status") is True or resp.get("status") == "true") and resp.get("data"):
                df = pd.DataFrame(
                    resp["data"],
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
                df = df.set_index("timestamp").sort_index()
                df = df.apply(pd.to_numeric, errors="coerce").dropna()
                _nifty_data_cache = df
                return df
            error_msg = resp.get("message", "Unknown error") if resp else "None response"
            raise ValueError(f"Empty response: {error_msg}")
        except Exception as e:
            if attempt == MAX_RETRIES:
                logger.warning(f"Nifty index data fetch failed: {e}")
                return pd.DataFrame()
            time.sleep(2 ** attempt)

    return pd.DataFrame()


def get_market_regime() -> dict:
    """
    Compute dynamic score threshold based on Nifty's position.
    Returns dict with: threshold, regime_label, nifty_price, nifty_sma200, nifty_rsi.

    Nifty above 200-SMA AND RSI > 50  →  Strong Bull  → Threshold = 4
    Nifty above 200-SMA BUT RSI < 50  →  Weak Bull    → Threshold = 5
    Nifty below 200-SMA AND RSI < 50  →  Bear Market  → Threshold = 6
    Nifty below 200-SMA AND RSI < 40  →  Crash Mode   → No trades at all (threshold=99)
    """
    global _market_regime_cache
    if _market_regime_cache is not None:
        return _market_regime_cache

    default = {"threshold": 4, "regime": "UNKNOWN", "price": 0.0,
               "sma200": 0.0, "rsi": 50.0, "no_trade": False}

    df = get_nifty_daily_data()
    if df.empty or len(df) < MIN_CANDLES:
        _market_regime_cache = default
        return default

    # Indicator computation is CPU-bound — fine in the main thread for index data
    df_e   = compute_indicators(df)
    lat    = df_e.iloc[-1]
    price  = float(lat["close"])
    sma200 = float(lat.get("SMA_200", price) or price)
    rsi    = float(lat.get("RSI_14", 50.0) or 50.0)

    above_200 = price > sma200

    if above_200 and rsi > 50:
        threshold = 4
        regime    = "STRONG BULL"
    elif above_200 and rsi <= 50:
        threshold = 5
        regime    = "WEAK BULL"
    elif not above_200 and rsi < 40:
        threshold = 99       # No trades in crash mode
        regime    = "CRASH MODE — NO ENTRIES"
    else:
        threshold = 6
        regime    = "BEAR MARKET"

    result = {
        "threshold": threshold,
        "regime":    regime,
        "price":     round(price, 2),
        "sma200":    round(sma200, 2),
        "rsi":       round(rsi, 1),
        "no_trade":  threshold == 99,
    }
    _market_regime_cache = result
    return result


def get_india_vix() -> float:
    """
    MEDIUM IMPACT FILTER: Fetch India VIX current value.
    Uses NSE's public API via a session-cookie approach.
    Falls back to 15.0 (safe neutral) on any failure.
    Cached for the run.
    """
    global _india_vix_cache
    if _india_vix_cache is not None:
        return _india_vix_cache

    vix = 15.0  # Safe default — below the 20 gate so won't block entries

    NSE_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.nseindia.com/",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            session = requests.Session()
            session.headers.update(NSE_HEADERS)

            # Establish cookie first
            session.get("https://www.nseindia.com", timeout=10)
            time.sleep(0.5)

            # Fetch all indices — India VIX is in this list
            resp = session.get(
                "https://www.nseindia.com/api/allIndices",
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("data", []):
                if item.get("index") == "INDIA VIX":
                    vix = float(item.get("last", 15.0))
                    _india_vix_cache = vix
                    return vix

            break  # Got response but VIX not found — use default

        except Exception as e:
            if attempt == MAX_RETRIES:
                logger.warning(f"India VIX fetch failed (using default {vix}): {e}")
            else:
                time.sleep(2 ** attempt)

    _india_vix_cache = vix
    return vix


def get_fii_net_flow() -> float:
    """
    HIGH IMPACT FILTER: Fetch FII/FPI net buy/sell from NSE's public report.
    Returns FII net value in crores (positive = net buy, negative = net sell).
    Falls back to 0.0 (neutral) on any failure — does not block entries.
    Cached for the run.

    NSE publishes this data daily after ~6 PM IST.
    During market hours, yesterday's data is used — still directionally valid.
    """
    global _fii_net_cache
    if _fii_net_cache is not None:
        return _fii_net_cache

    fii_net = 0.0  # Neutral default

    NSE_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.nseindia.com/",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            session = requests.Session()
            session.headers.update(NSE_HEADERS)

            # Establish cookie
            session.get("https://www.nseindia.com", timeout=10)
            time.sleep(0.5)

            resp = session.get(
                "https://www.nseindia.com/api/fiidiiTradeReact",
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            # Response is list of entries. We want the most recent FII equity net.
            # Structure: [{date, fii: {buyValue, sellValue, netValue}, dii: {...}}]
            if isinstance(data, list) and len(data) > 0:
                latest = data[0]
                fii_data = latest.get("fii", {}) or latest.get("FII", {})

                # Try multiple key patterns NSE uses
                net = (fii_data.get("netValue") or
                       fii_data.get("net_value") or
                       fii_data.get("NET VALUE") or 0.0)
                fii_net = float(net)
                _fii_net_cache = fii_net
                return fii_net

            break

        except Exception as e:
            if attempt == MAX_RETRIES:
                logger.warning(f"FII data fetch failed (using neutral 0.0): {e}")
            else:
                time.sleep(2 ** attempt)

    _fii_net_cache = fii_net
    return fii_net


def get_sector_scores() -> dict:
    """
    Fetch sector index data via SmartAPI and compute RSI + trend for each.
    Uses sector_index_token_map (populated from master file by fetch_token_map())
    rather than the hardcoded SECTOR_INDICES fallback table.
    Returns dict: sector_name → {rsi, above_ema20, trend, context}
    Cached for the run.
    """
    global _sector_scores_cache
    if _sector_scores_cache is not None:
        return _sector_scores_cache

    scores = {}
    # Use live tokens from master file; fall back to hardcoded if map is empty
    token_source = sector_index_token_map if sector_index_token_map else SECTOR_INDICES

    for sector, token in token_source.items():
        try:
            df = get_sector_index_data(sector, token)
            if df.empty or len(df) < 15:
                scores[sector] = {"rsi": 50, "above_ema20": True,
                                   "trend": "NEUTRAL", "context": f"{sector}: No data"}
                continue

            df_e     = compute_indicators(df)
            lat      = df_e.iloc[-1]
            rsi      = float(lat.get("RSI_14", 50) or 50)
            ema20    = float(lat.get("EMA_20", lat["close"]) or lat["close"])
            above_em = lat["close"] > ema20
            trend    = "BULLISH" if (above_em and rsi > 50) else (
                       "BEARISH" if (not above_em and rsi < 50) else "NEUTRAL"
            )

            scores[sector] = {
                "rsi":         round(rsi, 1),
                "above_ema20": above_em,
                "trend":       trend,
                "context":     f"{sector}: RSI={rsi:.0f}, {'↑' if above_em else '↓'}EMA20, {trend}",
            }
        except Exception as e:
            logger.debug(f"Sector score failed for {sector}: {e}")
            scores[sector] = {"rsi": 50, "above_ema20": True,
                               "trend": "NEUTRAL", "context": f"{sector}: Error"}

    _sector_scores_cache = scores
    return scores


def _get_sector_for_symbol(symbol: str) -> str:
    """Map a stock symbol to its sector name."""
    return SYMBOL_SECTOR.get(symbol, "Unknown")


def _build_sector_context_string(symbol: str, sector_scores: dict) -> str:
    """Build the sector context string to pass into the AI prompt."""
    sector = _get_sector_for_symbol(symbol)
    lines  = []

    # Stock's own sector first
    if sector in sector_scores:
        lines.append(sector_scores[sector]["context"])

    # Market-wide sector summary
    bull_sectors = [s for s, v in sector_scores.items() if v["trend"] == "BULLISH"]
    bear_sectors = [s for s, v in sector_scores.items() if v["trend"] == "BEARISH"]

    if bull_sectors:
        lines.append(f"Strong sectors: {', '.join(bull_sectors[:3])}")
    if bear_sectors:
        lines.append(f"Weak sectors: {', '.join(bear_sectors[:3])}")

    return " | ".join(lines) if lines else "Sector data unavailable"


# ─────────────────────────────────────────────────────────────────────────────
#  DUAL AI ENGINE  (Gemini + Groq)
# ─────────────────────────────────────────────────────────────────────────────
_gemini_client = None   # google.genai.Client (new unified SDK)
_groq_client   = None


def _init_gemini() -> bool:
    """
    Initialise the new Google Gen AI client (google-genai package).
    The old google-generativeai package reached end-of-life on Nov 30 2025.
    New pattern: genai.Client(api_key=...) → client.models.generate_content(...)
    """
    global _gemini_client
    if not GEMINI_AVAILABLE:
        return False
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False
    try:
        _gemini_client = google_genai.Client(api_key=api_key)
        return True
    except Exception as e:
        logger.warning(f"Gemini init failed: {e}")
        return False


def _init_groq() -> bool:
    global _groq_client
    if not GROQ_AVAILABLE:
        return False
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return False
    try:
        _groq_client = Groq(api_key=api_key)
        return True
    except Exception as e:
        logger.warning(f"Groq init failed: {e}")
        return False


def _build_ai_prompt(symbol: str, price: float, rsi: float, macd_hist: float,
                     ema20: float, sma200: float, atr: float, vol_ratio: float,
                     pattern: str, swing_high: float, swing_low: float,
                     formula_sl: float, sector_context: str,
                     market_regime: str) -> str:
    """
    Build the unified ~400-token AI prompt.
    Both Gemini and Groq receive the exact same prompt for fair consensus.
    """
    clean = symbol.replace("-EQ", "")

    above_200 = "YES" if price > sma200 else "NO"
    above_ema = "YES" if price > ema20 else "NO"
    macd_sign = "positive" if macd_hist > 0 else "negative"

    return f"""You are a professional Indian equity swing trader. Analyse {clean} for a 1-3 week swing trade.

STOCK: {clean}
PRICE: ₹{price:.2f}
RSI-14: {rsi:.1f}
MACD Histogram: {macd_hist:+.2f} ({macd_sign})
Above 200-SMA: {above_200} (SMA200=₹{sma200:.2f})
Price vs 20-EMA: {above_ema} (EMA20=₹{ema20:.2f})
ATR-14: ₹{atr:.2f} ({(atr/price*100):.1f}% of price)
Volume Ratio: {vol_ratio:.2f}x (vs 20-day avg)
Candlestick Pattern: {pattern or 'None detected'}
Swing High: ₹{swing_high:.2f if swing_high else 0}
Swing Low: ₹{swing_low:.2f if swing_low else 0}
Formula SL: ₹{formula_sl:.2f}
Sector Context: {sector_context}
Market Regime: {market_regime}

Reply ONLY in this exact JSON (no extra text, no markdown):
{{
  "fundamental_bias": "BULLISH|NEUTRAL|BEARISH",
  "technical_bias": "BULLISH|NEUTRAL|BEARISH",
  "overall_bias": "BULLISH|NEUTRAL|BEARISH",
  "confidence": 0.0,
  "suggested_sl": 0.0,
  "suggested_tp1": 0.0,
  "suggested_tp2": 0.0,
  "earnings_risk": false,
  "earnings_note": "",
  "sector_outlook": "BULLISH|NEUTRAL|BEARISH",
  "reasoning": "max 20 words"
}}"""


def _parse_ai_response(raw: str, symbol: str) -> Optional[dict]:
    """Parse and validate AI JSON response. Returns None on failure."""
    try:
        raw = raw.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)

        # Validate required keys
        required = ["overall_bias", "suggested_sl", "suggested_tp1", "suggested_tp2",
                    "confidence", "earnings_risk"]
        for key in required:
            if key not in data:
                raise ValueError(f"Missing key: {key}")

        return {
            "fundamental_bias": str(data.get("fundamental_bias", "NEUTRAL")).upper(),
            "technical_bias":   str(data.get("technical_bias",   "NEUTRAL")).upper(),
            "overall_bias":     str(data.get("overall_bias",      "NEUTRAL")).upper(),
            "confidence":       max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            "suggested_sl":     float(data.get("suggested_sl", 0.0)),
            "suggested_tp1":    float(data.get("suggested_tp1", 0.0)),
            "suggested_tp2":    float(data.get("suggested_tp2", 0.0)),
            "earnings_risk":    bool(data.get("earnings_risk", False)),
            "earnings_note":    str(data.get("earnings_note", ""))[:100],
            "sector_outlook":   str(data.get("sector_outlook", "NEUTRAL")).upper(),
            "reasoning":        str(data.get("reasoning", ""))[:100],
        }
    except Exception as e:
        logger.warning(f"[{symbol}] AI response parse error: {e} | raw: {raw[:100]}")
        return None


def _call_gemini(prompt: str, symbol: str) -> Optional[dict]:
    """
    Call Gemini 2.0 Flash via the new google-genai SDK and return parsed response.
    New SDK: client.models.generate_content() with types.GenerateContentConfig()
    """
    global _gemini_client
    if _gemini_client is None:
        if not _init_gemini():
            return None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            return _parse_ai_response(response.text, symbol)
        except Exception as e:
            logger.warning(f"[{symbol}] Gemini call error (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    return None


def _call_groq(prompt: str, symbol: str) -> Optional[dict]:
    """Call Groq Qwen3-32B and return parsed response or None."""
    if _groq_client is None:
        if not _init_groq():
            return None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = _groq_client.chat.completions.create(
                model="qwen/qwen3-32b",      # Qwen3 32B on Groq (replaces deprecated qwen-qwq-32b)
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,             # Groq-recommended for Qwen3 reasoning
                max_tokens=600,
                reasoning_format="hidden",   # Suppress <think> chain; return only JSON output
            )
            raw = completion.choices[0].message.content
            return _parse_ai_response(raw, symbol)
        except Exception as e:
            logger.warning(f"[{symbol}] Groq call error (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    return None


def _apply_sl_tp_safety(entry: float, ai_sl: float, ai_tp1: float,
                        ai_tp2: float, formula_sl: float) -> tuple[float, float, float]:
    """
    Apply safety rules to AI-suggested SL and TP.
    SL rules:
      - Cannot be more than 4% below entry (too wide)
      - Cannot be less than 1% below entry (too tight)
      - Cannot be above formula SL (formula is the floor)
    TP rules:
      - TP1 and TP2 cannot be more than 15% above entry
      - TP2 must be > TP1
    Returns (safe_sl, safe_tp1, safe_tp2).
    """
    # SL safety
    min_sl = entry * (1 - AI_SL_MAX_PCT)   # not more than 4% below
    max_sl = entry * (1 - AI_SL_MIN_PCT)   # not less than 1% below
    # AI SL must be below formula SL (formula is the floor)
    effective_max_sl = min(max_sl, formula_sl)

    if ai_sl <= 0 or ai_sl >= entry:
        safe_sl = formula_sl
    else:
        safe_sl = max(min_sl, min(effective_max_sl, ai_sl))

    # TP safety
    max_tp = entry * (1 + AI_TP_MAX_PCT)

    if ai_tp1 <= entry or ai_tp1 > max_tp:
        safe_tp1 = min(entry * 1.06, max_tp)   # default 6% target
    else:
        safe_tp1 = ai_tp1

    if ai_tp2 <= safe_tp1 or ai_tp2 > max_tp:
        safe_tp2 = min(entry * 1.12, max_tp)  # default 12% target
    else:
        safe_tp2 = ai_tp2

    return round(safe_sl, 2), round(safe_tp1, 2), round(safe_tp2, 2)


def get_dual_ai_analysis(symbol: str, price: float, rsi: float, macd_hist: float,
                         ema20: float, sma200: float, atr: float, vol_ratio: float,
                         pattern: str, swing_high: float, swing_low: float,
                         formula_sl: float, sector_context: str,
                         market_regime: str) -> dict:
    """
    Call both Gemini and Groq with identical context.
    Consensus logic:
      Both BULLISH  → ai_score = +2, use averaged SL/TP
      Both BEARISH  → ai_score = -3
      Disagree      → ai_score = 0  (neutral — no AI bonus)
    Returns structured dict with consensus result.
    """
    NEUTRAL_RESULT = {
        "ai_score":    0,
        "gemini_bias": "NEUTRAL",
        "groq_bias":   "NEUTRAL",
        "consensus":   "DISAGREE",
        "suggested_sl":  formula_sl,
        "suggested_tp1": 0.0,
        "suggested_tp2": 0.0,
        "earnings_risk": False,
        "earnings_note": "",
        "reasoning":     "AI unavailable",
        "confidence":    0.0,
    }

    prompt = _build_ai_prompt(
        symbol, price, rsi, macd_hist, ema20, sma200, atr, vol_ratio,
        pattern, swing_high or 0.0, swing_low or 0.0, formula_sl,
        sector_context, market_regime
    )

    gemini_result = _call_gemini(prompt, symbol)
    groq_result   = _call_groq(prompt, symbol)

    # If both failed
    if gemini_result is None and groq_result is None:
        return NEUTRAL_RESULT

    # Use whichever is available
    g_bias = gemini_result["overall_bias"] if gemini_result else "NEUTRAL"
    r_bias = groq_result["overall_bias"]   if groq_result   else "NEUTRAL"

    # Consensus evaluation
    both_bullish = (g_bias == "BULLISH" and r_bias == "BULLISH")
    both_bearish = (g_bias == "BEARISH" and r_bias == "BEARISH")

    if both_bullish:
        ai_score  = 2
        consensus = "BULLISH"
    elif both_bearish:
        ai_score  = -3
        consensus = "BEARISH"
    else:
        ai_score  = 0
        consensus = "DISAGREE"

    # Average SL/TP if both bullish and both gave valid suggestions
    if both_bullish and gemini_result and groq_result:
        raw_sl  = (gemini_result["suggested_sl"]  + groq_result["suggested_sl"])  / 2
        raw_tp1 = (gemini_result["suggested_tp1"] + groq_result["suggested_tp1"]) / 2
        raw_tp2 = (gemini_result["suggested_tp2"] + groq_result["suggested_tp2"]) / 2
    elif gemini_result:
        raw_sl, raw_tp1, raw_tp2 = (
            gemini_result["suggested_sl"],
            gemini_result["suggested_tp1"],
            gemini_result["suggested_tp2"],
        )
    elif groq_result:
        raw_sl, raw_tp1, raw_tp2 = (
            groq_result["suggested_sl"],
            groq_result["suggested_tp1"],
            groq_result["suggested_tp2"],
        )
    else:
        raw_sl, raw_tp1, raw_tp2 = formula_sl, 0.0, 0.0

    safe_sl, safe_tp1, safe_tp2 = _apply_sl_tp_safety(
        price, raw_sl, raw_tp1, raw_tp2, formula_sl
    )

    # Earnings risk — either AI flagging it is enough to trigger
    earnings_risk = bool(
        (gemini_result and gemini_result["earnings_risk"]) or
        (groq_result   and groq_result["earnings_risk"])
    )
    earnings_note = (
        (gemini_result and gemini_result.get("earnings_note")) or
        (groq_result   and groq_result.get("earnings_note"))   or ""
    )

    avg_confidence = (
        ((gemini_result["confidence"] if gemini_result else 0.0) +
         (groq_result["confidence"]   if groq_result   else 0.0)) /
        (2 if (gemini_result and groq_result) else 1)
    )

    reasoning = (
        (gemini_result and gemini_result.get("reasoning")) or
        (groq_result   and groq_result.get("reasoning"))   or ""
    )

    return {
        "ai_score":    ai_score,
        "gemini_bias": g_bias,
        "groq_bias":   r_bias,
        "consensus":   consensus,
        "suggested_sl":  safe_sl,
        "suggested_tp1": safe_tp1,
        "suggested_tp2": safe_tp2,
        "earnings_risk": earnings_risk,
        "earnings_note": earnings_note,
        "reasoning":     reasoning,
        "confidence":    round(avg_confidence, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  FULL STOCK ANALYSER
#  Architecture:
#    1. Data fetching — single-threaded (SmartAPI rate limit)
#    2. Indicator computation — parallel threads (CPU-bound, no I/O)
#    3. AI calls — sequential (external API rate limits)
# ─────────────────────────────────────────────────────────────────────────────
def analyse_stock(symbol: str, nifty_df: pd.DataFrame,
                  market_regime: dict, sector_scores: dict) -> Optional[dict]:
    """
    Full multi-timeframe + dual AI analysis for one symbol.

    Data fetch: single-threaded (sequential API calls)
    Scoring:    parallel threads via ThreadPoolExecutor
    AI calls:   sequential (rate-limited external APIs)
    """

    # ── 1. DATA FETCH (single-threaded) ──────────────────────────────────────
    df_daily  = get_daily_data(symbol)
    df_hourly = get_hourly_data(symbol)
    df_15min  = get_15min_data(symbol)
    df_weekly = get_weekly_data(symbol)

    if df_daily.empty or len(df_daily) < MIN_CANDLES:
        return None

    # ── 2. COMPUTE INDICATORS (parallel threads) ──────────────────────────────
    # compute_indicators operates on copies of DataFrames — thread-safe
    with ThreadPoolExecutor(max_workers=CALC_WORKERS) as pool:
        future_daily_e = pool.submit(compute_indicators, df_daily)
        future_weekly_e = pool.submit(compute_indicators, df_weekly) if not df_weekly.empty else None

        # Submit scoring tasks (hourly and 15m compute their own indicators internally)
        future_hourly  = pool.submit(score_hourly, df_hourly)
        future_15min   = pool.submit(score_15min,  df_15min)
        future_weekly_trend = pool.submit(check_weekly_trend, df_weekly) if not df_weekly.empty else None

        # Collect results
        df_daily_e = future_daily_e.result()
        score_h, sigs_h = future_hourly.result()
        score_m, sigs_m = future_15min.result()
        weekly_aligned, weekly_bias = (
            future_weekly_trend.result() if future_weekly_trend else (True, "UNKNOWN")
        )

    # ── 3. DAILY SCORE (uses pre-computed enriched df) ────────────────────────
    score_d, sigs_d = score_daily(df_daily_e)

    # ── 4. EXTRACT KEY VALUES ─────────────────────────────────────────────────
    latest = df_daily_e.iloc[-1]

    price      = float(latest["close"])
    rsi_val    = float(latest.get("RSI_14", 50.0) or 50.0)
    ema20      = float(latest.get("EMA_20", price) or price)
    atr        = float(latest.get("ATRr_14", price * 0.02) or price * 0.02)
    sma200     = float(latest.get("SMA_200", price) or price)
    macd_hist  = float(latest.get("MACDh_12_26_9", 0.0) or 0.0)
    vol_ratio  = float(latest.get("vol_ratio", 1.0) or 1.0)

    swing = find_swing_levels(df_daily_e)

    # ── 5. HIGH-IMPACT FILTERS (computed on enriched df — thread results) ─────

    # Volume confirmation
    vol_ok, vol_r = check_volume_confirmation(df_daily_e)

    # Relative strength vs Nifty
    rs_ok, rs_score = check_relative_strength(df_daily, nifty_df)

    # ── 6. MEDIUM-IMPACT FILTERS ──────────────────────────────────────────────

    # Weekly trend alignment (computed in parallel above)
    # Note: if weekly_aligned is False, we note it but it's a soft signal
    # (score penalty rather than hard block — ensures we don't lose too many setups)

    # ── 7. FORMULA SL (floor — AI cannot go above this) ──────────────────────
    # FIX #1: Only use swing high if it's ABOVE current price
    formula_sl = max(ema20 - atr * 1.5, price * 0.95)

    swing_high_valid = (
        swing["swing_high"]
        if (swing["swing_high"] and swing["swing_high"] > price)
        else None
    )
    formula_tp = swing_high_valid if swing_high_valid else (price + atr * 3)

    # ── 8. DETECT PRIMARY PATTERN (for AI context) ────────────────────────────
    patterns = detect_candlestick_patterns(df_daily_e)
    detected_pattern = next(
        (name for name, val in patterns.items() if val and "bearish" not in name),
        "None"
    )

    # ── 9. SECTOR CONTEXT STRING ──────────────────────────────────────────────
    sector          = _get_sector_for_symbol(symbol)
    sector_ctx_str  = _build_sector_context_string(symbol, sector_scores)
    regime_str      = f"{market_regime['regime']} (RSI={market_regime['rsi']})"

    # ── 10. DUAL AI ANALYSIS ──────────────────────────────────────────────────
    ai = get_dual_ai_analysis(
        symbol=symbol,
        price=price, rsi=rsi_val, macd_hist=macd_hist,
        ema20=ema20, sma200=sma200, atr=atr, vol_ratio=vol_ratio,
        pattern=detected_pattern,
        swing_high=swing["swing_high"], swing_low=swing["swing_low"],
        formula_sl=formula_sl,
        sector_context=sector_ctx_str,
        market_regime=regime_str,
    )

    # ── 11. FINAL SL / TP ─────────────────────────────────────────────────────
    final_sl  = ai["suggested_sl"] if ai["suggested_sl"] > 0 else formula_sl
    final_tp1 = ai["suggested_tp1"] if ai["suggested_tp1"] > price else formula_tp * 0.7
    final_tp2 = ai["suggested_tp2"] if ai["suggested_tp2"] > final_tp1 else formula_tp

    # Ensure SL is below price (sanity check)
    if final_sl >= price:
        final_sl = formula_sl

    rr_ratio = ((final_tp2 - price) / (price - final_sl)) if (price - final_sl) > 0 else 0.0

    # ── 12. TOTAL SCORE ───────────────────────────────────────────────────────
    total_score = score_d + ai["ai_score"]

    if score_h >= 2:
        total_score += 1
        sigs_h.append("H1: Multi-signal bonus")

    if score_m >= 1:   # FIX #4: 15m now contributes +1
        total_score += 1

    # Weekly misalignment penalty
    if not weekly_aligned and weekly_bias == "BEARISH":
        total_score -= 1
        sigs_d.append(f"⚠ Weekly trend BEARISH (penalty -1)")

    # Volume filter penalty (not a hard block — just score reduction)
    if not vol_ok:
        total_score -= 1
        sigs_d.append(f"⚠ Low volume {vol_r:.2f}x (penalty -1)")

    # Relative strength bonus
    if rs_ok:
        total_score += 1
        sigs_d.append(f"RS vs Nifty: {rs_score:.2f}x ↑")
    else:
        sigs_d.append(f"RS vs Nifty: {rs_score:.2f}x (lagging)")

    # ── 13. POSITION SIZING (uses LIVE ledger capital — passed from main) ─────
    # FIX #5: Risk sizing deferred to enter_trade() which has live capital

    all_signals = (
        sigs_d +
        [f"AI Gemini: {ai['gemini_bias']} | Groq: {ai['groq_bias']} | {ai['consensus']}"] +
        ([f"⚠ EARNINGS: {ai['earnings_note']}"] if ai["earnings_risk"] else []) +
        [f"H1:{s}" for s in sigs_h] +
        [f"15m:{s}" for s in sigs_m]
    )

    return {
        "symbol":          symbol,
        "sector":          sector,
        "price":           round(price, 2),
        "rsi":             round(rsi_val, 1),
        "ema20":           round(ema20, 2),
        "sma200":          round(sma200, 2),
        "atr":             round(atr, 2),
        "vol_ratio":       round(vol_r, 2),
        "rs_score":        round(rs_score, 4),
        "weekly_bias":     weekly_bias,
        "score":           total_score,
        "score_daily":     score_d,
        "score_hourly":    score_h,
        "score_15m":       score_m,
        "ai_score":        ai["ai_score"],
        "gemini_bias":     ai["gemini_bias"],
        "groq_bias":       ai["groq_bias"],
        "ai_consensus":    ai["consensus"],
        "ai_confidence":   ai["confidence"],
        "ai_reasoning":    ai["reasoning"],
        "earnings_risk":   ai["earnings_risk"],
        "earnings_note":   ai["earnings_note"],
        "stop_loss":       round(final_sl, 2),
        "take_profit_1":   round(final_tp1, 2),
        "take_profit_2":   round(final_tp2, 2),
        "take_profit":     round(final_tp2, 2),   # backward compat for ledger
        "rr_ratio":        round(rr_ratio, 2),
        "signals":         all_signals,
        "timestamp":       datetime.now().isoformat(),
        # qty and position_value computed in enter_trade() with live capital
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PAPER TRADING LEDGER
# ─────────────────────────────────────────────────────────────────────────────
class PaperLedger:
    """
    Persistent paper trading engine.
    FIX #2: enter_trade() works on a dict copy — never mutates scan result.
    FIX #5: Position sizing uses live self.capital.
    NEW:     Trailing SL logic in check_exits().
    NEW:     Portfolio heat check (max 2 positions per sector).
    """

    def __init__(self):
        self.capital         = PAPER_CAPITAL
        self.open_positions  : dict  = {}
        self.trade_history   : list  = []
        self.total_realised  : float = 0.0
        self._load_state()

    def _load_state(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)
                self.capital        = state.get("capital", PAPER_CAPITAL)
                self.open_positions = state.get("open_positions", {})
                self.trade_history  = state.get("trade_history", [])
                self.total_realised = state.get("total_realised", 0.0)
                console.print(
                    f"[dim]  Loaded ledger: ₹{self.capital:,.2f} capital, "
                    f"{len(self.open_positions)} open positions[/]"
                )
            except Exception as e:
                logger.warning(f"Could not load ledger: {e} — starting fresh")

    def save_state(self):
        try:
            state = {
                "capital":        self.capital,
                "open_positions": self.open_positions,
                "trade_history":  self.trade_history,
                "total_realised": self.total_realised,
                "last_updated":   datetime.now().isoformat(),
            }
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Ledger save failed: {e}")

    def _append_trade_csv(self, trade: dict):
        try:
            df = pd.DataFrame([trade])
            df.to_csv(TRADES_CSV, mode="a", header=not TRADES_CSV.exists(), index=False)
        except Exception as e:
            logger.warning(f"CSV log failed: {e}")

    def _sector_count(self) -> dict:
        """Count open positions per sector."""
        counts = {}
        for pos in self.open_positions.values():
            sec = pos.get("sector", "Unknown")
            counts[sec] = counts.get(sec, 0) + 1
        return counts

    def enter_trade(self, analysis: dict) -> Optional[dict]:
        """
        Paper BUY entry.
        FIX #2: Works on a copy of analysis dict.
        FIX #5: Uses live self.capital for position sizing.
        NEW: Sector heat check.
        """
        sym = analysis["symbol"]

        if sym in self.open_positions:
            return None

        if len(self.open_positions) >= MAX_OPEN_POSITIONS:
            return None

        # Portfolio heat check — no more than MAX_SECTOR_EXPOSURE in same sector
        sector_counts = self._sector_count()
        stock_sector  = analysis.get("sector", "Unknown")
        if sector_counts.get(stock_sector, 0) >= MAX_SECTOR_EXPOSURE:
            logger.debug(f"[{sym}] Sector limit reached for {stock_sector}")
            return None

        if analysis["rr_ratio"] < 1.5:
            return None

        # FIX #2: Work on a copy — never mutate the scan result
        a = dict(analysis)

        price = a["price"]
        sl    = a["stop_loss"]

        if sl <= 0 or sl >= price:
            return None

        risk_per_share = price - sl
        if risk_per_share <= 0:
            return None

        # FIX #5: Use LIVE capital for position sizing
        risk_amount    = self.capital * RISK_PER_TRADE
        qty            = int(risk_amount / risk_per_share)

        if qty <= 0:
            return None

        position_value = qty * price

        # Cap single position at 40% of available capital
        if position_value > self.capital * 0.4:
            qty            = int(self.capital * 0.4 / price)
            position_value = qty * price

        if position_value <= 0 or position_value > self.capital:
            return None

        position = {
            "symbol":         sym,
            "sector":         stock_sector,
            "entry_price":    price,
            "qty":            qty,
            "stop_loss":      sl,
            "stop_loss_orig": sl,       # For trailing SL tracking
            "take_profit_1":  a.get("take_profit_1", a.get("take_profit", price * 1.06)),
            "take_profit_2":  a.get("take_profit_2", a.get("take_profit", price * 1.12)),
            "take_profit":    a.get("take_profit_2", a.get("take_profit", price * 1.12)),
            "entry_score":    a["score"],
            "gemini_bias":    a.get("gemini_bias", "NEUTRAL"),
            "groq_bias":      a.get("groq_bias", "NEUTRAL"),
            "ai_consensus":   a.get("ai_consensus", "NEUTRAL"),
            "entry_time":     datetime.now().isoformat(),
            "position_value": round(position_value, 2),
            "atr":            a.get("atr", price * 0.02),
            "signals":        a.get("signals", [])[:5],
            "tp1_triggered":  False,    # Track partial exit at TP1
        }

        self.open_positions[sym] = position
        self.capital -= position_value
        self.save_state()
        return position

    def check_exits(self, current_prices: dict) -> list:
        """
        Check all open positions against current prices.
        NEW: Trailing SL — activates after 1.5× ATR gain, trails to breakeven+.
        Handles partial exit at TP1 (50% off), full exit at TP2.
        """
        closed = []

        for sym, pos in list(self.open_positions.items()):
            cmp = current_prices.get(sym)
            if not cmp:
                continue

            entry  = pos["entry_price"]
            atr    = pos.get("atr", entry * 0.02)
            sl     = pos["stop_loss"]
            tp1    = pos["take_profit_1"]
            tp2    = pos["take_profit_2"]

            # ── TRAILING SL ───────────────────────────────────────────────────
            # Activate after 1.5× ATR gain; trail to keep ≥ breakeven+
            gain = cmp - entry
            if gain >= atr * 1.5:
                # Trail: SL = max(current SL, entry + 0.2× ATR)
                trail_sl = entry + atr * 0.2
                if trail_sl > sl:
                    self.open_positions[sym]["stop_loss"] = round(trail_sl, 2)
                    sl = trail_sl

            # ── CHECK EXIT CONDITIONS ─────────────────────────────────────────
            exit_reason = None
            exit_qty    = pos["qty"]

            if cmp <= sl:
                exit_reason = "STOP LOSS HIT"
            elif not pos.get("tp1_triggered") and cmp >= tp1:
                # Partial exit at TP1 (50% of position)
                exit_reason = "TP1 PARTIAL EXIT (50%)"
                exit_qty = max(1, pos["qty"] // 2)
                # Update position: reduce qty, raise SL to entry (cost-free)
                new_qty = pos["qty"] - exit_qty
                if new_qty > 0:
                    self.open_positions[sym]["qty"]           = new_qty
                    self.open_positions[sym]["position_value"] = round(new_qty * entry, 2)
                    self.open_positions[sym]["stop_loss"]      = entry   # breakeven
                    self.open_positions[sym]["tp1_triggered"]  = True
                    self.save_state()
                    # Record partial close as a trade
                    pnl = (cmp - entry) * exit_qty
                    self.total_realised += pnl
                    partial_record = {
                        **pos,
                        "qty":         exit_qty,
                        "exit_price":  cmp,
                        "exit_time":   datetime.now().isoformat(),
                        "exit_reason": exit_reason,
                        "pnl":         round(pnl, 2),
                        "pnl_pct":     round(((cmp - entry) / entry) * 100, 2),
                    }
                    self.trade_history.append(partial_record)
                    self._append_trade_csv(partial_record)
                    closed.append(partial_record)
                    continue  # position still open

                exit_reason = "TP1 FULL CLOSE"  # edge case: qty=1

            elif cmp >= tp2:
                exit_reason = "TARGET TP2 HIT"

            if exit_reason:
                pnl     = (cmp - entry) * exit_qty
                pnl_pct = ((cmp - entry) / entry) * 100

                trade_record = {
                    **pos,
                    "qty":         exit_qty,
                    "exit_price":  cmp,
                    "exit_time":   datetime.now().isoformat(),
                    "exit_reason": exit_reason,
                    "pnl":         round(pnl, 2),
                    "pnl_pct":     round(pnl_pct, 2),
                }

                self.capital        += pos["position_value"] + pnl
                self.total_realised += pnl
                self.trade_history.append(trade_record)
                del self.open_positions[sym]
                self._append_trade_csv(trade_record)
                closed.append(trade_record)

        if closed:
            self.save_state()
        return closed

    def unrealised_pnl(self, current_prices: dict) -> float:
        total = 0.0
        for sym, pos in self.open_positions.items():
            cmp = current_prices.get(sym)
            if cmp:
                total += (cmp - pos["entry_price"]) * pos["qty"]
        return total

    def win_rate(self) -> float:
        if not self.trade_history:
            return 0.0
        wins = sum(1 for t in self.trade_history if t.get("pnl", 0) > 0)
        return (wins / len(self.trade_history)) * 100

    def summary_stats(self) -> dict:
        if not self.trade_history:
            return {}
        pnls = [t["pnl"] for t in self.trade_history]
        return {
            "total_trades": len(pnls),
            "winners":      sum(1 for p in pnls if p > 0),
            "losers":       sum(1 for p in pnls if p < 0),
            "win_rate":     round(self.win_rate(), 1),
            "total_pnl":    round(sum(pnls), 2),
            "avg_win":      round(np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0, 2),
            "avg_loss":     round(np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0, 2),
            "best_trade":   round(max(pnls), 2),
            "worst_trade":  round(min(pnls), 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  RICH TERMINAL DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
def display_header():
    console.print()
    console.print(Panel(
        Align.center(
            "[bold white]NIFTY 50 AI SWING TRADING AGENT  v4.4[/]\n"
            "[dim]Murphy Framework  •  Dual AI Consensus (Gemini + Groq)  •  Dynamic Regime[/]\n"
            "[dim cyan]Volume Filter  •  FII Gate  •  Relative Strength  •  VIX Gate  •  Weekly Alignment[/]"
        ),
        border_style="bold cyan",
        padding=(1, 4),
    ))
    console.print(
        f"[dim]  Run started: {datetime.now().strftime('%A, %d %b %Y  %H:%M:%S')}  |  "
        f"Output: {OUTPUT_DIR}[/]"
    )
    console.print()


def display_market_context(regime: dict, vix: float, fii_net: float,
                           sector_scores: dict):
    """Display market-wide context panel before stock scan."""
    console.print(Rule("[bold white] MARKET CONTEXT [/]", style="dim"))

    # Regime
    reg_col = (
        "green" if "BULL" in regime["regime"]
        else "red" if "CRASH" in regime["regime"] or "BEAR" in regime["regime"]
        else "yellow"
    )
    console.print(
        f"  Nifty Regime : [{reg_col}]{regime['regime']}[/]  "
        f"| Threshold: [bold]{regime['threshold']}[/]  "
        f"| Nifty: ₹{regime['price']:,.2f}  "
        f"| SMA200: ₹{regime['sma200']:,.2f}  "
        f"| RSI: {regime['rsi']}"
    )

    # VIX
    vix_col = "red" if vix > VIX_MAX_FOR_ENTRY else "green"
    vix_gate = "[red]BLOCKED[/]" if vix > VIX_MAX_FOR_ENTRY else "[green]OPEN[/]"
    console.print(f"  India VIX    : [{vix_col}]{vix:.2f}[/]  | Entry Gate: {vix_gate}")

    # FII
    fii_col  = "green" if fii_net > 0 else "red"
    fii_gate = (
        f"[red]BLOCKED (FII sold ₹{abs(fii_net):,.0f} Cr)[/]"
        if fii_net < -FII_SELL_THRESHOLD_CR
        else f"[green]OPEN[/]"
    )
    console.print(
        f"  FII Net Flow : [{fii_col}]₹{fii_net:+,.0f} Cr[/]  | Entry Gate: {fii_gate}"
    )

    # Sector heat
    bull_s = [s.replace("NIFTY ", "") for s, v in sector_scores.items() if v["trend"] == "BULLISH"]
    bear_s = [s.replace("NIFTY ", "") for s, v in sector_scores.items() if v["trend"] == "BEARISH"]
    if bull_s:
        console.print(f"  Hot Sectors  : [green]{', '.join(bull_s)}[/]")
    if bear_s:
        console.print(f"  Cold Sectors : [red]{', '.join(bear_s)}[/]")

    console.print()


def display_scan_results(results: list, threshold: int):
    """Rich table of all analysed stocks sorted by score."""
    console.print(Rule("[bold white] ANALYSIS RESULTS [/]", style="cyan"))

    table = Table(
        box=box.MINIMAL_DOUBLE_HEAD,
        show_header=True,
        header_style="bold white on grey15",
        padding=(0, 1),
        expand=True,
    )

    table.add_column("Symbol",     style="bold",       width=13)
    table.add_column("Sector",     style="dim",        width=10)
    table.add_column("CMP (₹)",    justify="right",    width=10)
    table.add_column("RSI",        justify="center",   width=6)
    table.add_column("Vol×",       justify="center",   width=6)
    table.add_column("RS",         justify="center",   width=6)
    table.add_column("Score",      justify="center",   width=7)
    table.add_column("D/H/M/AI",   justify="center",   width=10)
    table.add_column("Gemini",     justify="center",   width=8)
    table.add_column("Groq",       justify="center",   width=8)
    table.add_column("Consensus",  justify="center",   width=10)
    table.add_column("SL (₹)",     justify="right",    width=10)
    table.add_column("TP1 (₹)",    justify="right",    width=10)
    table.add_column("TP2 (₹)",    justify="right",    width=10)
    table.add_column("R:R",        justify="center",   width=6)
    table.add_column("Weekly",     justify="center",   width=8)
    table.add_column("Earnings",   justify="center",   width=9)

    def _bias_badge(bias: str) -> str:
        if bias == "BULLISH":  return "[green]▲ BULL[/]"
        if bias == "BEARISH":  return "[red]▼ BEAR[/]"
        return "[dim]◆ NTRL[/]"

    def _consensus_badge(c: str) -> str:
        if c == "BULLISH":  return "[bold green]✓ AGREE ↑[/]"
        if c == "BEARISH":  return "[bold red]✗ AGREE ↓[/]"
        return "[yellow]≠ SPLIT[/]"

    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        score = r["score"]

        if score >= threshold:
            row_style = "bold green"
        elif score == threshold - 1:
            row_style = "yellow"
        elif score <= 0:
            row_style = "dim red"
        else:
            row_style = None

        score_str = (
            f"[bold green]{score}[/]" if score >= threshold else
            f"[yellow]{score}[/]" if score > 0 else f"[red]{score}[/]"
        )
        rr    = r["rr_ratio"]
        rr_s  = f"[green]{rr:.1f}x[/]" if rr >= 2 else (
                f"[yellow]{rr:.1f}x[/]" if rr >= 1.5 else f"[red]{rr:.1f}x[/]")

        vol_s = (
            f"[green]{r['vol_ratio']:.1f}[/]" if r["vol_ratio"] >= VOLUME_CONFIRM_RATIO
            else f"[red]{r['vol_ratio']:.1f}[/]"
        )
        rs_s  = (
            f"[green]{r['rs_score']:.2f}[/]" if r["rs_score"] >= 1.0
            else f"[red]{r['rs_score']:.2f}[/]"
        )
        weekly_s  = (
            "[green]BULL[/]" if r["weekly_bias"] == "BULLISH" else
            "[red]BEAR[/]"  if r["weekly_bias"] == "BEARISH" else
            "[dim]—[/]"
        )
        earn_s = "[red]⚠ YES[/]" if r.get("earnings_risk") else "[dim]No[/]"

        table.add_row(
            r["symbol"].replace("-EQ", ""),
            r.get("sector", "")[:9],
            f"{r['price']:,.2f}",
            f"{r['rsi']:.1f}",
            vol_s,
            rs_s,
            score_str,
            f"{r['score_daily']}/{r['score_hourly']}/{r['score_15m']}/{r['ai_score']:+d}",
            _bias_badge(r.get("gemini_bias", "NEUTRAL")),
            _bias_badge(r.get("groq_bias", "NEUTRAL")),
            _consensus_badge(r.get("ai_consensus", "NEUTRAL")),
            f"{r['stop_loss']:,.2f}",
            f"{r.get('take_profit_1', r['take_profit']):,.2f}",
            f"{r.get('take_profit_2', r['take_profit']):,.2f}",
            rr_s,
            weekly_s,
            earn_s,
            style=row_style if score >= threshold else None,
        )

    console.print(table)


def display_trade_executions(new_trades: list):
    if not new_trades:
        return
    console.print(Rule("[bold green] PAPER TRADE ENTRIES [/]", style="green"))
    for t in new_trades:
        sym = t["symbol"].replace("-EQ", "")
        console.print(Panel(
            f"[bold green]  ▶ BUY  {sym:12}  ×{t['qty']} shares @ ₹{t['entry_price']:,.2f}[/]\n"
            f"  Sector     : {t.get('sector','?')}  |  Consensus: {t.get('ai_consensus','?')}\n"
            f"  Stop Loss  : ₹{t['stop_loss']:,.2f}  |  TP1 (50%): ₹{t.get('take_profit_1', t['take_profit']):,.2f}\n"
            f"  TP2 (full) : ₹{t.get('take_profit_2', t['take_profit']):,.2f}  |  Risk: ₹{(t['entry_price']-t['stop_loss'])*t['qty']:,.2f}\n"
            f"  Pos Value  : ₹{t['position_value']:,.2f}  |  Score: {t['entry_score']}\n"
            f"  Signals    : {' | '.join(t.get('signals',[])[:3])}",
            border_style="green",
            title=f"[white]PAPER TRADE — {datetime.now().strftime('%H:%M:%S')}[/]"
        ))


def display_exits(closed_trades: list):
    if not closed_trades:
        return
    console.print(Rule("[bold yellow] TRADE EXITS [/]", style="yellow"))
    for t in closed_trades:
        sym  = t["symbol"].replace("-EQ", "")
        pnl  = t["pnl"]
        col  = "green" if pnl > 0 else "red"
        icon = "✓" if pnl > 0 else "✗"
        console.print(
            f"  [{col}]{icon} {sym:10}  {t['exit_reason']:24}  "
            f"Entry ₹{t['entry_price']:,.2f} → Exit ₹{t['exit_price']:,.2f}  "
            f"P&L: ₹{pnl:+,.2f} ({t['pnl_pct']:+.2f}%)[/{col}]"
        )


def display_portfolio(ledger: PaperLedger, current_prices: dict):
    console.print(Rule("[bold cyan] PORTFOLIO STATUS [/]", style="cyan"))

    unreal = ledger.unrealised_pnl(current_prices)
    net_equity   = ledger.capital + sum(
        p["position_value"] for p in ledger.open_positions.values()
    ) + unreal
    total_return = ((net_equity - PAPER_CAPITAL) / PAPER_CAPITAL) * 100

    stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    stats_table.add_column(style="dim", width=22)
    stats_table.add_column(style="bold")

    stats_table.add_row("Starting Capital",  f"₹{PAPER_CAPITAL:>12,.2f}")
    stats_table.add_row("Available Cash",    f"₹{ledger.capital:>12,.2f}")
    stats_table.add_row("Open Positions",    str(len(ledger.open_positions)))
    stats_table.add_row("Unrealised P&L",    f"[{'green' if unreal>=0 else 'red'}]₹{unreal:>+12,.2f}[/]")
    stats_table.add_row("Realised P&L",      f"[{'green' if ledger.total_realised>=0 else 'red'}]₹{ledger.total_realised:>+12,.2f}[/]")
    stats_table.add_row("Net Equity",        f"₹{net_equity:>12,.2f}")
    stats_table.add_row("Total Return",      f"[{'green' if total_return>=0 else 'red'}]{total_return:>+11.2f}%[/]")

    stats = ledger.summary_stats()
    if stats:
        stats_table.add_row("", "")
        stats_table.add_row("Total Trades",   str(stats["total_trades"]))
        stats_table.add_row("Win Rate",       f"{stats['win_rate']}%  (W:{stats['winners']} L:{stats['losers']})")
        stats_table.add_row("Best Trade",     f"[green]₹{stats['best_trade']:+,.2f}[/]")
        stats_table.add_row("Worst Trade",    f"[red]₹{stats['worst_trade']:+,.2f}[/]")

    console.print(stats_table)

    if ledger.open_positions:
        console.print("\n[bold]  Open Positions:[/]")
        pos_table = Table(box=box.SIMPLE, show_header=True,
                          header_style="bold on grey15")
        pos_table.add_column("Symbol",     width=12)
        pos_table.add_column("Sector",     width=10)
        pos_table.add_column("Qty",        justify="right")
        pos_table.add_column("Entry",      justify="right")
        pos_table.add_column("CMP",        justify="right")
        pos_table.add_column("SL",         justify="right")
        pos_table.add_column("TP1",        justify="right")
        pos_table.add_column("TP2",        justify="right")
        pos_table.add_column("Unreal P&L", justify="right")
        pos_table.add_column("Days",       justify="right")
        pos_table.add_column("Consensus",  justify="center")

        for sym, pos in ledger.open_positions.items():
            cmp      = current_prices.get(sym, pos["entry_price"])
            upnl     = (cmp - pos["entry_price"]) * pos["qty"]
            upnl_pct = ((cmp - pos["entry_price"]) / pos["entry_price"]) * 100
            days_held = (datetime.now() - datetime.fromisoformat(pos["entry_time"])).days
            pnl_col  = "green" if upnl >= 0 else "red"

            pos_table.add_row(
                sym.replace("-EQ", ""),
                pos.get("sector", "")[:9],
                str(pos["qty"]),
                f"₹{pos['entry_price']:,.2f}",
                f"₹{cmp:,.2f}",
                f"₹{pos['stop_loss']:,.2f}",
                f"₹{pos.get('take_profit_1', pos['take_profit']):,.2f}",
                f"₹{pos.get('take_profit_2', pos['take_profit']):,.2f}",
                f"[{pnl_col}]₹{upnl:+,.2f} ({upnl_pct:+.1f}%)[/]",
                str(days_held),
                pos.get("ai_consensus", "—"),
            )
        console.print(pos_table)


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE DAILY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def save_daily_summary(results: list, new_trades: list, closed_trades: list,
                       ledger: PaperLedger, market_ctx: dict):
    try:
        summary = {
            "run_date":          datetime.now().isoformat(),
            "market_regime":     market_ctx.get("regime"),
            "score_threshold":   market_ctx.get("threshold"),
            "india_vix":         market_ctx.get("vix"),
            "fii_net":           market_ctx.get("fii_net"),
            "symbols_scanned":   len(results),
            "qualified_setups":  len([r for r in results
                                      if r["score"] >= market_ctx.get("threshold", 4)]),
            "trades_entered":    len(new_trades),
            "trades_closed":     len(closed_trades),
            "top_picks":         sorted(results, key=lambda x: x["score"], reverse=True)[:5],
            "ledger_stats":      ledger.summary_stats(),
            "capital":           ledger.capital,
            "realised_pnl":      ledger.total_realised,
        }
        with open(SUMMARY_FILE, "w") as f:
            json.dump(summary, f, indent=2)
        console.print(f"[dim]  Summary saved → {SUMMARY_FILE}[/]")
    except Exception as e:
        logger.warning(f"Summary save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global smartApi

    display_header()

    # ── Broker Init ──────────────────────────────────────────────────────────
    console.print(Rule("[bold white] BROKER SETUP [/]", style="dim"))
    smartApi = init_broker()
    if smartApi is None:
        console.print("[bold red]  SmartAPI initialisation failed. Cannot proceed.[/]")
        sys.exit(1)
    fetch_token_map()

    # ── AI Init ──────────────────────────────────────────────────────────────
    console.print(Rule("[bold white] AI ENGINES [/]", style="dim"))
    console.print("  [dim]Using new google-genai SDK (google-generativeai is EOL)[/]")

    gemini_ok = False
    if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
        gemini_ok = _init_gemini()
        status = "[green]✓ Active[/]" if gemini_ok else "[yellow]✗ Init failed[/]"
    else:
        status = "[dim]Not configured (GEMINI_API_KEY missing)[/]"
    console.print(f"  Gemini 2.0 Flash : {status}")

    groq_ok = False
    if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
        groq_ok = _init_groq()
        status = "[green]✓ Active[/]" if groq_ok else "[yellow]✗ Init failed[/]"
    else:
        status = "[dim]Not configured (GROQ_API_KEY missing)[/]"
    console.print(f"  Groq Qwen3-32B   : {status}")

    if gemini_ok and groq_ok:
        console.print("  [green]✓ Dual AI consensus active — both AIs required to agree[/]")
    elif gemini_ok or groq_ok:
        console.print("  [yellow]⚠ Single AI mode — add second API key for full consensus[/]")
    else:
        console.print("  [dim]No AI configured — running on technical signals only[/]")

    # ── Load Ledger ──────────────────────────────────────────────────────────
    ledger = PaperLedger()

    # ── Market-Wide Data (fetched once, cached) ───────────────────────────────
    console.print()
    console.print(Rule("[bold white] MARKET PRE-SCAN [/]", style="dim"))
    console.print("  [dim]Fetching Nifty index, India VIX, FII data, sector scores...[/]")

    market_regime  = get_market_regime()
    india_vix      = get_india_vix()
    fii_net        = get_fii_net_flow()
    sector_scores  = get_sector_scores()
    nifty_df       = get_nifty_daily_data()

    display_market_context(market_regime, india_vix, fii_net, sector_scores)

    # ── Hard blocks ───────────────────────────────────────────────────────────
    if market_regime["no_trade"]:
        console.print(Panel(
            f"[bold red]  CRASH MODE ACTIVE — {market_regime['regime']}\n"
            f"  No new entries. Monitoring open positions only.[/]",
            border_style="red"
        ))
        # Still check exits on open positions
        current_prices = {}
        closed_trades = ledger.check_exits(current_prices)
        display_exits(closed_trades)
        display_portfolio(ledger, current_prices)
        return

    if india_vix > VIX_MAX_FOR_ENTRY:
        console.print(
            f"[bold red]  VIX GATE: India VIX={india_vix:.1f} > {VIX_MAX_FOR_ENTRY} — "
            f"No new entries today.[/]"
        )

    fii_blocked = fii_net < -FII_SELL_THRESHOLD_CR
    if fii_blocked:
        console.print(
            f"[bold red]  FII GATE: FII sold ₹{abs(fii_net):,.0f} Cr — "
            f"No new entries today.[/]"
        )

    # ── Stock Scan ───────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold white] STOCK SCAN [/]", style="dim"))

    results          = []
    current_prices   = {}
    symbols_to_scan  = [s for s in NIFTY_50_SYMBOLS if s in token_map]

    console.print(
        f"  Scanning [bold cyan]{len(symbols_to_scan)}[/] symbols  "
        f"| Regime threshold: [bold cyan]{market_regime['threshold']}[/]  "
        f"| Parallel compute: [bold cyan]{CALC_WORKERS} threads[/]\n"
    )

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, style="cyan", complete_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[dim]{task.fields[status]}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            "Analysing stocks...",
            total=len(symbols_to_scan),
            status=""
        )

        for sym in symbols_to_scan:
            progress.update(
                task,
                description=f"  Scanning [bold]{sym.replace('-EQ','')}[/]",
                status=""
            )
            try:
                result = analyse_stock(
                    sym, nifty_df, market_regime, sector_scores
                )
                if result:
                    results.append(result)
                    current_prices[sym] = result["price"]
            except Exception as e:
                logger.error(f"[{sym}] Unhandled error: {traceback.format_exc()}")

            progress.advance(task, 1)

    if not results:
        console.print("[bold red]No valid results — check connectivity and credentials.[/]")
        return

    # ── Display Results ───────────────────────────────────────────────────────
    console.print()
    display_scan_results(results, market_regime["threshold"])

    # ── Check Exits ───────────────────────────────────────────────────────────
    closed_trades = ledger.check_exits(current_prices)
    display_exits(closed_trades)

    # ── Execute New Trades ────────────────────────────────────────────────────
    new_trades = []

    # Hard gates: skip entries if VIX or FII blocked
    entries_allowed = not (india_vix > VIX_MAX_FOR_ENTRY or fii_blocked)

    if entries_allowed:
        qualified = [
            r for r in results
            if r["score"] >= market_regime["threshold"]
            and not r.get("earnings_risk")         # skip earnings events
        ]

        for r in sorted(qualified, key=lambda x: (x["score"], x["rr_ratio"]), reverse=True):
            pos = ledger.enter_trade(r)
            if pos:
                new_trades.append(pos)
    else:
        qualified = []

    console.print()
    if new_trades:
        display_trade_executions(new_trades)
    else:
        reason = ""
        if india_vix > VIX_MAX_FOR_ENTRY:
            reason = f"VIX={india_vix:.1f} gate active"
        elif fii_blocked:
            reason = f"FII sell gate active"
        elif not qualified:
            reason = f"No setups met threshold {market_regime['threshold']}"
        else:
            reason = "All rejected (positions full / sector limit / low R:R)"
        console.print(f"[yellow]  No entries — {reason}[/]")

    # ── Portfolio Status ──────────────────────────────────────────────────────
    console.print()
    display_portfolio(ledger, current_prices)

    # ── Save Summary ──────────────────────────────────────────────────────────
    console.print()
    market_ctx = {**market_regime, "vix": india_vix, "fii_net": fii_net}
    save_daily_summary(results, new_trades, closed_trades, ledger, market_ctx)

    # ── Final Line ────────────────────────────────────────────────────────────
    console.print()
    console.print(Rule(style="dim"))
    console.print(
        f"[dim]  v4.4 complete — {len(results)} scanned | "
        f"{len([r for r in results if r['score'] >= market_regime['threshold']])} qualified | "
        f"{len(new_trades)} entries | "
        f"{len(closed_trades)} exits | "
        f"Regime: {market_regime['regime']} | "
        f"VIX: {india_vix:.1f} | "
        f"Output: {OUTPUT_DIR}[/]"
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user. State saved.[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error: {e}[/]")
        logger.error(traceback.format_exc())
        sys.exit(1)
