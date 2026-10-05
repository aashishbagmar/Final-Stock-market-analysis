"""
AI Stock Analyst - Cloud-ready Streamlit app
Production optimizations: caching, env vars, safe fallbacks, lightweight modes.
"""

# --- Stdlib ---
import os
import re
import sys
import time
import hashlib
import json
import logging
import warnings
from datetime import datetime
from threading import Lock
from typing import Dict, List, Tuple, Optional

# --- Third-party ---
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# Detect whether this file is being executed by Streamlit or as a standalone script.
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    IN_STREAMLIT = get_script_run_ctx() is not None
except Exception:
    IN_STREAMLIT = False

# Provide a minimal session_state substitute when running outside Streamlit so
# functions from app.py that touch st.session_state do not crash.
if not IN_STREAMLIT:
    class _MockSessionState:
        def __init__(self):
            self._data = {}

        def get(self, key, default=None):
            return self._data.get(key, default)

        def __getitem__(self, key):
            return self._data[key]

        def __setitem__(self, key, value):
            self._data[key] = value

        def setdefault(self, key, default=None):
            return self._data.setdefault(key, default)

    st.session_state = _MockSessionState()


from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from yfinance.exceptions import YFRateLimitError
except Exception:
    YFRateLimitError = None

# --- Global env defaults ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

if load_dotenv:
    # Load local .env for development only; no override to keep real env vars priority.
    load_dotenv(override=False)

# --- Config ---
# Backtesting override: allow LSTM to train for more epochs when run as a script.
os.environ.setdefault("LSTM_MAX_EPOCHS", "15")

DEFAULT_CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "7200"))
CACHE_TTL_STOCK = int(os.getenv("CACHE_TTL_STOCK", "7200"))
CACHE_TTL_NEWS = int(os.getenv("CACHE_TTL_NEWS", "7200"))
CACHE_TTL_FEATURES = int(os.getenv("CACHE_TTL_FEATURES", "7200"))
CACHE_TTL_MODELS = int(os.getenv("CACHE_TTL_MODELS", "7200"))
CACHE_TTL_SENTIMENT = int(os.getenv("CACHE_TTL_SENTIMENT", "7200"))
CACHE_TTL_INFO = int(os.getenv("CACHE_TTL_INFO", "7200"))
MAX_ARTICLES_DEFAULT = int(os.getenv("MAX_NEWS_ARTICLES", "12"))
MAX_LSTM_EPOCHS = int(os.getenv("LSTM_MAX_EPOCHS", "10"))
LIGHTWEIGHT_ENV = os.getenv("LIGHTWEIGHT_MODE", "").lower() in ("1", "true", "yes")
DEBUG_FETCH_ENV = os.getenv("DEBUG_FETCH", "").lower() in ("1", "true", "yes")
YF_MIN_INTERVAL = float(os.getenv("YF_MIN_INTERVAL", "6.0"))
YF_MAX_RETRIES = int(os.getenv("YF_MAX_RETRIES", "1"))
YF_TIMEOUT = int(os.getenv("YF_TIMEOUT", "12"))
YF_COOLDOWN_SECONDS = int(os.getenv("YF_COOLDOWN_SECONDS", "90"))

TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-^]{0,12}$")

# --- Streamlit page config / CSS (only when running as a Streamlit app) ---
if IN_STREAMLIT:
    st.set_page_config(
        page_title="AI Stock Analyst",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- CSS (Bloomberg terminal + responsive tweaks) ---
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

*, html, body { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1,h2,h3,h4, .mono { font-family: 'IBM Plex Mono', monospace !important; }

.stApp { background: #080b10; color: #c9d1d9; }
.main .block-container { padding-top: 1.5rem; max-width: 1300px; }

.app-header { border-bottom: 1px solid #1c2333; padding-bottom: 1.2rem; margin-bottom: 1.5rem; }
.app-title { font-family: 'IBM Plex Mono', monospace; font-size: 1.9rem; font-weight: 700; color: #58a6ff; letter-spacing: 3px; text-transform: uppercase; }
.app-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #3d4f6e; letter-spacing: 4px; text-transform: uppercase; margin-top: 2px; }

.step-row { display: flex; align-items: center; gap: 10px; padding: 10px 16px; border-radius: 6px; margin: 5px 0; background: #0d1117; border: 1px solid #1c2333; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; transition: all 0.3s; }
.step-row.active { background: #0d1e33; border-color: #1f6feb; color: #58a6ff; }
.step-row.done { background: #0d1f17; border-color: #196c3b; color: #3fb950; }
.step-row.error { background: #1f0d0d; border-color: #6e1313; color: #f85149; }
.step-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.dot-wait  { background: #3d4f6e; }
.dot-active{ background: #1f6feb; box-shadow: 0 0 8px #1f6feb; animation: blink 1s infinite; }
.dot-done  { background: #3fb950; }
.dot-error { background: #f85149; }

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.signal-box { border-radius: 10px; padding: 32px 20px; text-align: center; font-family: 'IBM Plex Mono', monospace; }
.signal-BUY  { background:#051a0f; border:2px solid #3fb950; }
.signal-SELL { background:#1a0505; border:2px solid #f85149; }
.signal-HOLD { background:#111108; border:2px solid #d29922; }
.signal-word-BUY  { font-size:3.6rem; font-weight:700; color:#3fb950; letter-spacing:10px; }
.signal-word-SELL { font-size:3.6rem; font-weight:700; color:#f85149; letter-spacing:10px; }
.signal-word-HOLD { font-size:3.6rem; font-weight:700; color:#d29922; letter-spacing:10px; }
.signal-reason { color:#8b949e; margin-top:10px; font-size:0.85rem; }

.metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin:12px 0; }
.mcard { background:#0d1117; border:1px solid #1c2333; border-radius:8px; padding:14px 16px; }
.mcard-label { font-size:0.68rem; color:#3d4f6e; letter-spacing:2px; text-transform:uppercase; font-family:'IBM Plex Mono',monospace; }
.mcard-value { font-size:1.25rem; font-weight:600; color:#e6edf3; font-family:'IBM Plex Mono',monospace; margin-top:4px; }
.mcard-sub   { font-size:0.72rem; color:#8b949e; margin-top:2px; }

.news-item { border-left: 3px solid #1c2333; padding: 8px 14px; margin: 6px 0; font-size: 0.83rem; background: #0d1117; border-radius: 0 6px 6px 0; }
.news-pos { border-left-color: #3fb950; }
.news-neg { border-left-color: #f85149; }
.news-neu { border-left-color: #d29922; }

.model-row { display:grid; grid-template-columns: 160px 1fr 1fr 1fr; gap:8px; padding:10px 14px; border-radius:6px; margin:4px 0; background:#0d1117; border:1px solid #1c2333; font-family:'IBM Plex Mono',monospace; font-size:0.82rem; align-items:center; }
.model-row.header { color:#3d4f6e; font-size:0.7rem; letter-spacing:2px; background:transparent; border-color:transparent; }

.conf-bar-wrap { background:#1c2333; border-radius:4px; height:10px; overflow:hidden; margin-top:6px; }
.conf-bar-fill { height:100%; border-radius:4px; background:linear-gradient(90deg,#1f6feb,#58a6ff); }

.sent-bar-wrap { display:flex; height:14px; border-radius:6px; overflow:hidden; margin:8px 0; }

.live-badge { display:inline-block; background:#1a1f2e; border:1px solid #1f6feb; color:#58a6ff; font-family:'IBM Plex Mono',monospace; font-size:0.65rem; letter-spacing:3px; padding:2px 10px; border-radius:20px; }
.live-dot { display:inline-block; width:6px;height:6px; border-radius:50%; background:#3fb950; margin-right:5px; animation:blink 1.4s infinite; vertical-align:middle; }

.stButton > button { background: #1f6feb; color: #ffffff; border: none; border-radius: 6px; font-family: 'IBM Plex Mono', monospace; font-weight: 600; letter-spacing: 2px; padding: 0.55rem 1.8rem; width: 100%; transition: background 0.2s; }
.stButton > button:hover { background: #388bfd; }
.stTextInput > div > div > input { background: #0d1117; border: 1px solid #1c2333; border-radius: 6px; color: #e6edf3; font-family: 'IBM Plex Mono', monospace; font-size: 1rem; }
section[data-testid="stSidebar"] { background: #0d1117; }

@media (max-width: 900px) {
    .app-title { font-size: 1.4rem; letter-spacing: 2px; }
    .app-sub { font-size: 0.6rem; letter-spacing: 2px; }
    .model-row { grid-template-columns: 1fr; gap: 6px; }
    .signal-word-BUY, .signal-word-SELL, .signal-word-HOLD { font-size: 2.6rem; letter-spacing: 6px; }
    .main .block-container { padding: 1rem 0.8rem; }
}
</style>
""",
        unsafe_allow_html=True,
    )

# --- Pipeline steps ---
STEPS = [
    ("01", "User Input"),
    ("02", "Fetch Live Stock Data"),
    ("03", "Fetch Live News"),
    ("04", "Sentiment Analysis (VADER)"),
    ("05", "Feature Engineering"),
    ("06", "Train Linear Regression"),
    ("07", "Train Random Forest"),
    ("08", "Train LSTM"),
    ("09", "Ensemble Predictions"),
    ("10", "Generate Signal"),
]


def is_cloud_env() -> bool:
    return any(
        os.getenv(name)
        for name in ("RENDER", "RAILWAY_ENVIRONMENT", "STREAMLIT_SERVER_HEADLESS")
    )


MODEL_WARMUP = os.getenv("MODEL_WARMUP", "true").lower() == "true" and not is_cloud_env()


def get_secret(key: str) -> str:
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


def sanitize_ticker(raw: str) -> str:
    value = (raw or "").strip().upper()
    if not value:
        return ""
    if not TICKER_RE.match(value):
        return ""
    return value


def build_ticker_candidates(ticker: str) -> List[str]:
    cleaned = ticker.strip().upper().replace(" ", "")
    candidates = [cleaned]

    if cleaned.endswith(".NS"):
        candidates.append(cleaned.replace(".NS", ".BO"))
        candidates.append(cleaned.replace(".NS", ""))
    elif cleaned.endswith(".BO"):
        candidates.append(cleaned.replace(".BO", ".NS"))
        candidates.append(cleaned.replace(".BO", ""))
    else:
        if not cleaned.startswith("^"):
            candidates.append(f"{cleaned}.NS")
            candidates.append(f"{cleaned}.BO")

    # Add a lowercase version as a final fallback for Yahoo quirks.
    candidates.append(cleaned.lower())

    # Deduplicate while preserving order.
    seen = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._lock = Lock()
        self._last_call = 0.0
        self._cooldown_until = 0.0

    def wait(self) -> float:
        with self._lock:
            now = time.monotonic()
            if self._cooldown_until > now:
                time.sleep(self._cooldown_until - now)
                now = time.monotonic()
            elapsed = now - self._last_call
            remaining = self.min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_call = time.monotonic()
            return max(0.0, remaining)

    def cooldown(self, seconds: float) -> None:
        with self._lock:
            now = time.monotonic()
            self._cooldown_until = max(self._cooldown_until, now + seconds)


@st.cache_resource(show_spinner=False)
def get_rate_limiter() -> RateLimiter:
    return RateLimiter(YF_MIN_INTERVAL)


def is_rate_limit_error(err: Exception) -> bool:
    if YFRateLimitError and isinstance(err, YFRateLimitError):
        return True
    msg = str(err).lower()
    return "rate limit" in msg or "too many requests" in msg or "429" in msg


def make_run_key(ticker: str, inputs: Dict, effective_lightweight: bool, effective_skip_lstm: bool) -> str:
    payload = {
        "ticker": ticker,
        "horizon": inputs.get("horizon"),
        "max_articles": inputs.get("max_articles"),
        "lstm_epochs": inputs.get("lstm_epochs"),
        "skip_lstm": effective_skip_lstm,
        "lightweight": effective_lightweight,
        "use_cache": inputs.get("use_cache"),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        raise RuntimeError(
            "TensorFlow is not installed. Install it or enable 'Skip LSTM' in the sidebar."
        )


@st.cache_resource(show_spinner=False)
def get_scaler_template() -> MinMaxScaler:
    return MinMaxScaler()


def new_scaler() -> MinMaxScaler:
    try:
        return clone(get_scaler_template())
    except Exception:
        return MinMaxScaler()
@st.cache_resource(show_spinner=False)
def get_lstm_model_factory():
    def build(look_back: int, units_1: int, units_2: int):
        tf = get_tf()
        Sequential = tf.keras.models.Sequential
        LSTM = tf.keras.layers.LSTM
        Dense = tf.keras.layers.Dense
        Dropout = tf.keras.layers.Dropout
        model = Sequential(
            [
                LSTM(units_1, return_sequences=True, input_shape=(look_back, 1)),
                Dropout(0.2),
                LSTM(units_2),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="huber")
        return model

    return build


def get_transformers():
    # Retained for compatibility; BERT replaced by VADER for cloud memory safety.
    return None, None, None


def validate_ohlcv(df: pd.DataFrame, min_rows: int = 30) -> Tuple[bool, pd.DataFrame, str]:
    if df is None or df.empty:
        return False, df, "empty"
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(df.columns):
        return False, df, "missing_ohlcv"
    df = df[required]
    if df.dropna(how="all").empty:
        return False, df, "all_nan"
    if df["Close"].dropna().empty:
        return False, df, "no_close"
    if len(df) < min_rows:
        return False, df, "not_enough_rows"
    return True, df.dropna(), "ok"


def log_fetch_event(event: str, details: Dict, logs: List[Dict]) -> None:
    if not st.session_state.get("debug_fetch"):
        return
    logger = logging.getLogger("stock_fetch")
    payload = {"event": event, **details}
    logger.info("stock_fetch %s", payload)
    logs.append(payload)


@st.cache_data(ttl=CACHE_TTL_INFO, show_spinner=False)
def get_ticker_info_cached(ticker: str) -> Dict:
    session = get_yf_session()
    t = yf.Ticker(ticker, session=session)
    info: Dict[str, object] = {}

    try:
        fast = getattr(t, "fast_info", None)
        if fast:
            info = {
                "name": getattr(fast, "shortName", None) or getattr(fast, "longName", None) or ticker,
                "currency": getattr(fast, "currency", "USD") or "USD",
                "exchange": getattr(fast, "exchange", "") or "",
                "sector": "N/A",
                "market_cap": getattr(fast, "marketCap", 0) or 0,
            }
    except Exception:
        info = {}

    if not info:
        try:
            raw = t.info
            info = {
                "name": raw.get("longName") or raw.get("shortName") or ticker,
                "currency": raw.get("currency", "USD"),
                "exchange": raw.get("exchange", ""),
                "sector": raw.get("sector", "N/A"),
                "market_cap": raw.get("marketCap", 0),
            }
        except Exception:
            info = {}

    if not info:
        info = {"name": ticker, "currency": "USD", "exchange": "", "sector": "N/A", "market_cap": 0}
    return info


def data_signature(df: pd.DataFrame) -> str:
    tail = df.tail(120)
    payload = pd.util.hash_pandas_object(tail, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def articles_signature(articles: List[Dict]) -> str:
    if not articles:
        return "empty"
    payload = "|".join(
        f"{a.get('title','')}|{a.get('date','')}|{a.get('source','')}" for a in articles
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@st.cache_resource(show_spinner=False)
def get_requests_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=1,
        backoff_factor=0.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@st.cache_resource(show_spinner=False)
def get_yf_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    adapter = HTTPAdapter(max_retries=Retry(total=0), pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@st.cache_resource(show_spinner=False)
def load_bert_pipeline():
    """Load VADER sentiment analyzer — lightweight, no model download required."""
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def warmup_models() -> None:
    # Background warmups are disabled for Streamlit Cloud stability.
    return


@st.cache_data(ttl=CACHE_TTL_STOCK, show_spinner=False)
def fetch_live_stock_cached(ticker: str) -> Tuple[pd.DataFrame, Dict]:
    return fetch_live_stock_uncached(ticker)


def fetch_live_stock_uncached(ticker: str) -> Tuple[pd.DataFrame, Dict]:
    # Robust yfinance fetch with cooldown, limited retries, and validation.
    logs: List[Dict] = []
    if st.session_state.get("debug_fetch"):
        st.session_state["fetch_logs"] = logs
    st.session_state["using_cached_data"] = False
    st.session_state["rate_limited"] = False

    session = get_yf_session()
    limiter = get_rate_limiter()
    is_cloud = is_cloud_env()

    candidates = build_ticker_candidates(ticker)
    log_fetch_event("candidate_tickers", {"input": ticker, "candidates": candidates}, logs)

    # Cloud-safe history length to reduce payload and throttling.
    period = "1y" if is_cloud else "2y"
    # Max one retry to avoid retry storms.
    max_retries = 1 + min(max(YF_MAX_RETRIES, 0), 1)
    timeout = YF_TIMEOUT

    ok = False
    last_error = ""
    for cand in candidates:
        for attempt in range(1, max_retries + 1):
            log_fetch_event("download_attempt", {"ticker": cand, "attempt": attempt}, logs)

            df = pd.DataFrame()
            try:
                limiter.wait()
                try:
                    df = yf.download(
                        cand,
                        progress=False,
                        auto_adjust=False,
                        group_by="column",
                        threads=False,
                        timeout=timeout,
                        session=session,
                        period=period,
                        interval="1d",
                    )
                except TypeError:
                    # Newer yfinance versions don't accept timeout/session/group_by
                    df = yf.download(
                        cand,
                        progress=False,
                        auto_adjust=False,
                        period=period,
                        interval="1d",
                    )
            except Exception as e:
                last_error = str(e)
                if is_rate_limit_error(e):
                    st.session_state["rate_limited"] = True
                    limiter.cooldown(YF_COOLDOWN_SECONDS)
                log_fetch_event(
                    "download_error",
                    {"ticker": cand, "error": last_error},
                    logs,
                )
                df = pd.DataFrame()

            ok, cleaned, reason = validate_ohlcv(df)
            log_fetch_event(
                "download_validate",
                {
                    "ticker": cand,
                    "rows": len(df) if df is not None else 0,
                    "columns": list(df.columns) if df is not None else [],
                    "status": reason,
                },
                logs,
            )
            if ok:
                df = cleaned

            if not ok and not st.session_state.get("rate_limited") and attempt == max_retries:
                try:
                    limiter.wait()
                    history = yf.Ticker(cand, session=session).history(
                        period=period,
                        interval="1d",
                        auto_adjust=False,
                        actions=False,
                        repair=True,
                        timeout=timeout,
                    )
                except TypeError:
                    history = yf.Ticker(cand, session=session).history(
                        period=period,
                        interval="1d",
                        auto_adjust=False,
                        actions=False,
                        repair=True,
                    )
                except Exception as e:
                    last_error = str(e)
                    if is_rate_limit_error(e):
                        st.session_state["rate_limited"] = True
                        limiter.cooldown(YF_COOLDOWN_SECONDS)
                    history = pd.DataFrame()
                    log_fetch_event(
                        "history_error",
                        {"ticker": cand, "error": last_error},
                        logs,
                    )

                ok, cleaned, reason = validate_ohlcv(history)
                log_fetch_event(
                    "history_validate",
                    {
                        "ticker": cand,
                        "rows": len(history) if history is not None else 0,
                        "columns": list(history.columns) if history is not None else [],
                        "status": reason,
                    },
                    logs,
                )
                if ok:
                    df = cleaned

            if ok:
                break

            if st.session_state.get("rate_limited"):
                break

        if ok:
            ticker = cand
            break
        if st.session_state.get("rate_limited"):
            break

    if not ok:
        cached = st.session_state.get("last_good_data", {}).get(ticker)
        if cached is not None:
            st.session_state["using_cached_data"] = True
            log_fetch_event("fallback_cached", {"ticker": ticker}, logs)
            return cached
        hint = f" Last error: {last_error}" if last_error else ""
        raise ValueError(f"No price data found for '{ticker}'. Check the ticker symbol or network.{hint}")

    info = get_ticker_info_cached(ticker)
    st.session_state.setdefault("last_good_data", {})[ticker] = (df, info)
    return df, info


@st.cache_data(ttl=CACHE_TTL_NEWS, show_spinner=False)
def fetch_live_news_cached(query: str, api_key: str, n: int) -> List[Dict]:
    return fetch_live_news_uncached(query, api_key, n)


def fetch_live_news_uncached(query: str, api_key: str, n: int) -> List[Dict]:
    session = get_requests_session()
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={requests.utils.quote(query)}"
        "&sortBy=publishedAt&language=en"
        f"&pageSize={n}&apiKey={api_key}"
    )
    resp = session.get(url, timeout=12)
    if resp.status_code == 429:
        raise ConnectionError("NewsAPI rate limit reached. Try again later.")
    if resp.status_code != 200:
        try:
            msg = resp.json().get("message", "")
        except Exception:
            msg = ""
        raise ConnectionError(f"NewsAPI error {resp.status_code}. {msg}")
    articles = resp.json().get("articles", [])
    return [
        {
            "title": a.get("title", ""),
            "desc": a.get("description") or "",
            "url": a.get("url", ""),
            "date": (a.get("publishedAt") or "")[:10],
            "source": a.get("source", {}).get("name", ""),
        }
        for a in articles
        if a.get("title")
    ]


@st.cache_data(
    ttl=CACHE_TTL_FEATURES,
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda df: data_signature(df)},
)
def engineer_features_cached(df: pd.DataFrame, sig: str) -> pd.DataFrame:
    return engineer_features_uncached(df)


def engineer_features_uncached(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["Close"]].copy()
    d["MA7"] = d["Close"].rolling(7).mean()
    d["MA21"] = d["Close"].rolling(21).mean()
    d["MA50"] = d["Close"].rolling(50).mean()
    d["Std7"] = d["Close"].rolling(7).std()
    d["Pct1"] = d["Close"].pct_change(1)
    d["Pct5"] = d["Close"].pct_change(5)
    d["Pct20"] = d["Close"].pct_change(20)
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    return d.dropna()


@st.cache_data(
    ttl=CACHE_TTL_MODELS,
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda df: data_signature(df)},
)
def train_linear_regression_cached(feat: pd.DataFrame, horizon: int, sig: str) -> Dict:
    return train_linear_regression_uncached(feat, horizon)


def train_linear_regression_uncached(feat: pd.DataFrame, horizon: int) -> Dict:
    X = feat.copy()
    y = X["Close"].shift(-horizon).dropna()
    X = X.iloc[: len(y)]

    split = int(len(X) * 0.85)
    scaler = new_scaler()
    Xtr = scaler.fit_transform(X.iloc[:split])
    ytr = y.iloc[:split]
    Xte = scaler.transform(X.iloc[split:])
    yte = y.iloc[split:]

    model = LinearRegression().fit(Xtr, ytr)
    mae = round(mean_absolute_error(yte, model.predict(Xte)), 4)
    pred = float(model.predict(scaler.transform(feat.iloc[[-1]]))[0])
    return {"model": "Linear Regression", "predicted": round(pred, 2), "mae": mae}


@st.cache_data(
    ttl=CACHE_TTL_MODELS,
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda df: data_signature(df)},
)
def train_random_forest_cached(feat: pd.DataFrame, horizon: int, sig: str) -> Dict:
    return train_random_forest_uncached(feat, horizon)


def train_random_forest_uncached(feat: pd.DataFrame, horizon: int) -> Dict:
    X = feat.copy()
    y = X["Close"].shift(-horizon).dropna()
    X = X.iloc[: len(y)]

    split = int(len(X) * 0.85)
    scaler = new_scaler()
    Xtr = scaler.fit_transform(X.iloc[:split])
    ytr = y.iloc[:split]
    Xte = scaler.transform(X.iloc[split:])
    yte = y.iloc[split:]

    n_estimators = 80 if is_cloud_env() else 160
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Xtr, ytr)
    mae = round(mean_absolute_error(yte, model.predict(Xte)), 4)
    pred = float(model.predict(scaler.transform(feat.iloc[[-1]]))[0])
    return {"model": "Random Forest", "predicted": round(pred, 2), "mae": mae}


def configure_tensorflow(lightweight: bool) -> None:
    try:
        tf = get_tf()
        tf.get_logger().setLevel("ERROR")
        if lightweight:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass


@st.cache_data(
    ttl=CACHE_TTL_MODELS,
    show_spinner=False,
    hash_funcs={pd.DataFrame: lambda df: data_signature(df)},
)
def train_lstm_cached(
    df: pd.DataFrame,
    horizon: int,
    epochs: int,
    lightweight: bool,
    sig: str,
) -> Dict:
    return train_lstm_uncached(df, horizon, epochs, lightweight)


def train_lstm_uncached(df: pd.DataFrame, horizon: int, epochs: int, lightweight: bool) -> Dict:
    configure_tensorflow(lightweight)
    tf = get_tf()
    tf.keras.backend.clear_session()

    prices = df["Close"].values.astype(np.float32)
    scaler = new_scaler()
    scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    look_back = 20 if lightweight else 30
    look_back = min(look_back, max(10, len(scaled) // 10))

    xs, ys = [], []
    for i in range(look_back, len(scaled) - horizon):
        xs.append(scaled[i - look_back : i])
        ys.append(scaled[i + horizon - 1])
    if not xs:
        raise ValueError("Not enough data for LSTM training.")

    xs = np.array(xs).reshape(-1, look_back, 1)
    ys = np.array(ys)

    split = int(len(xs) * 0.85)
    xtr, xte = xs[:split], xs[split:]
    ytr, yte = ys[:split], ys[split:]

    units_1 = 32 if lightweight else 48
    units_2 = 16 if lightweight else 24
    batch_size = 16 if lightweight else 32
    epochs = max(3, min(epochs, MAX_LSTM_EPOCHS))

    EarlyStopping = tf.keras.callbacks.EarlyStopping
    model = get_lstm_model_factory()(look_back, units_1, units_2)

    es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=0)
    model.fit(
        xtr,
        ytr,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )

    test_pred = scaler.inverse_transform(model.predict(xte, verbose=0))
    test_true = scaler.inverse_transform(yte.reshape(-1, 1))
    mae = round(float(mean_absolute_error(test_true, test_pred)), 4)

    seq = scaled[-look_back:].copy()
    for _ in range(horizon):
        inp = seq[-look_back:].reshape(1, look_back, 1)
        nxt = model.predict(inp, verbose=0)[0][0]
        seq = np.append(seq, nxt)
    pred = float(scaler.inverse_transform([[seq[-1]]])[0][0])

    tf.keras.backend.clear_session()
    return {"model": "LSTM", "predicted": round(pred, 2), "mae": mae}


def run_bert_sentiment(articles: List[Dict]) -> Dict:
    if not articles:
        return {
            "articles": [],
            "counts": {"positive": 0, "neutral": 1, "negative": 0},
            "pct_pos": 0,
            "pct_neu": 100,
            "pct_neg": 0,
            "score": 0.5,
            "overall": "Neutral",
        }

    sia = load_bert_pipeline()
    scored = []
    for art in articles:
        text = (art.get("title", "") + ". " + art.get("desc", ""))[:512]
        try:
            if sia is not None:
                compound = sia.polarity_scores(text)["compound"]
                if compound >= 0.05:
                    sentiment = "positive"
                elif compound <= -0.05:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                conf = round(min(1.0, abs(compound) + 0.5), 4)
            else:
                sentiment, conf = "neutral", 0.5
        except Exception:
            sentiment, conf = "neutral", 0.5
        scored.append({**art, "sentiment": sentiment, "confidence": conf})

    total = len(scored) or 1
    cnts = {k: sum(1 for s in scored if s["sentiment"] == k) for k in ("positive", "neutral", "negative")}
    score = (cnts["positive"] * 1.0 + cnts["neutral"] * 0.5) / total

    return {
        "articles": scored,
        "counts": cnts,
        "pct_pos": round(cnts["positive"] / total * 100, 1),
        "pct_neu": round(cnts["neutral"] / total * 100, 1),
        "pct_neg": round(cnts["negative"] / total * 100, 1),
        "score": round(score, 4),
        "overall": "Positive" if score > 0.6 else ("Negative" if score < 0.4 else "Neutral"),
    }


@st.cache_data(ttl=CACHE_TTL_SENTIMENT, show_spinner=False)
def run_bert_sentiment_cached(articles: List[Dict], sig: str) -> Dict:
    return run_bert_sentiment(articles)


def ensemble(lr: Dict, rf: Dict, lstm: Dict, current: float) -> Dict:
    def inv(x):
        return 1.0 / (x + 1e-6)

    wlr, wrf, wlstm = inv(lr["mae"]), inv(rf["mae"]), inv(lstm["mae"])
    total_w = wlr + wrf + wlstm
    final = (wlr * lr["predicted"] + wrf * rf["predicted"] + wlstm * lstm["predicted"]) / total_w

    preds = np.array([lr["predicted"], rf["predicted"], lstm["predicted"]])
    cv = preds.std() / (preds.mean() + 1e-9)
    confidence = round(max(0.0, min(1.0, 1 - cv)) * 100, 1)
    pct = round((final - current) / current * 100, 2)

    return {
        "final": round(final, 2),
        "pct_change": pct,
        "confidence": confidence,
        "current": round(current, 2),
    }


def make_signal(ens: Dict, sent: Dict) -> Dict:
    ss = sent["score"]
    g = ens["pct_change"]
    if ss > 0.55 and g > 1.0:
        signal = "BUY"
    elif ss < 0.45 and g < -1.0:
        signal = "SELL"
    else:
        signal = "HOLD"

    reasons = {
        "BUY": f"Bullish sentiment ({ss:.0%}) combined with +{g:.1f}% projected growth",
        "SELL": f"Bearish sentiment ({ss:.0%}) combined with {g:.1f}% projected decline",
        "HOLD": f"Mixed signals — sentiment {ss:.0%}, projected change {g:+.1f}%",
    }
    return {"signal": signal, "reason": reasons[signal]}


def fmt(val: float, currency: str = "USD") -> str:
    sym = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£"}.get(currency, currency + " ")
    return f"{sym}{val:,.2f}"


def volatility(df: pd.DataFrame) -> Tuple[float, str]:
    v = df["Close"].pct_change().std() * (252**0.5) * 100
    lbl = "Low 🟢" if v < 20 else ("Medium 🟡" if v < 40 else "High 🔴")
    return round(v, 1), lbl


def render_pipeline(states: Dict, placeholder) -> None:
    html = "<div style='margin-top:8px;'>"
    for idx, (num, label) in enumerate(STEPS):
        state = states.get(idx, "wait")
        dot_cls = {"wait": "dot-wait", "active": "dot-active", "done": "dot-done", "error": "dot-error"}[state]
        row_cls = {"wait": "", "active": " active", "done": " done", "error": " error"}[state]
        icon = {"wait": "○", "active": "◉", "done": "✓", "error": "✗"}[state]
        html += (
            f"<div class='step-row{row_cls}'>"
            f"<span class='step-dot {dot_cls}'></span>"
            f"<span style='color:#3d4f6e;'>STEP {num}</span>"
            f"<span style='flex:1'>{label}</span>"
            f"<span>{icon}</span>"
            "</div>"
        )
    html += "</div>"
    placeholder.markdown(html, unsafe_allow_html=True)


def main() -> None:
    warmup_models()

    st.markdown(
        """
        <div class='app-header'>
            <div class='app-title'>📡 AI STOCK ANALYST</div>
            <div class='app-sub'>LIVE DATA · BERT · LINEAR REGRESSION · RANDOM FOREST · LSTM · ENSEMBLE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "run_inputs" not in st.session_state:
        st.session_state["run_inputs"] = None
    if "run_key" not in st.session_state:
        st.session_state["run_key"] = None
    if "results_by_key" not in st.session_state:
        st.session_state["results_by_key"] = {}

    with st.sidebar:
        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#3d4f6e;letter-spacing:3px;'>CONFIGURATION</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        news_api_key = get_secret("NEWS_API_KEY")
        if news_api_key:
            st.success("NewsAPI: configured")
        else:
            st.warning("NewsAPI: missing. Set NEWS_API_KEY in env.")

        with st.form("config_form", clear_on_submit=False):
            ticker_input = st.text_input(
                "Ticker",
                value=st.session_state.get("ticker_input", ""),
                placeholder="Enter ticker e.g. AAPL / TSLA / TCS.NS",
            )
            horizon = st.slider("Forecast Horizon (days)", 5, 30, 7)
            max_articles = st.slider("News Articles to Fetch", 5, 20, MAX_ARTICLES_DEFAULT)
            lstm_epochs = st.slider("LSTM Epochs", 3, 12, 6)
            lightweight = st.toggle(
                "Lightweight mode (cloud-safe)",
                value=(LIGHTWEIGHT_ENV or is_cloud_env()),
            )
            skip_lstm = st.toggle("Skip LSTM (faster)", value=is_cloud_env())
            use_cache = st.toggle("Use cache (faster)", value=True)
            debug_fetch = st.toggle("Debug fetch logs", value=DEBUG_FETCH_ENV)
            submit = st.form_submit_button("RUN →")

        st.markdown("---")
        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#3d4f6e;letter-spacing:3px;'>QUICK TICKERS</div>",
            unsafe_allow_html=True,
        )
        quick = ["AAPL", "TSLA", "TCS.NS", "RELIANCE.NS", "INFY.NS", "MSFT", "NVDA", "HDFCBANK.NS"]
        cols = st.columns(2)
        for i, t in enumerate(quick):
            if cols[i % 2].button(t, key=f"q_{t}"):
                st.session_state["ticker_input"] = t

        st.markdown("---")
        st.markdown(
            "<div style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#3d4f6e;'>PIPELINE PROGRESS</div>",
            unsafe_allow_html=True,
        )
        pipeline_ph = st.empty()

        if st.session_state.get("warmup_future"):
            future = st.session_state["warmup_future"]
            if future.done():
                st.caption("Model warmup: ready")
            else:
                st.caption("Model warmup: running")

    render_pipeline({}, pipeline_ph)

    if submit:
        st.session_state["run_inputs"] = {
            "ticker": ticker_input,
            "horizon": horizon,
            "max_articles": max_articles,
            "lstm_epochs": lstm_epochs,
            "skip_lstm": skip_lstm,
            "lightweight": lightweight,
            "use_cache": use_cache,
        }
        st.session_state["debug_fetch"] = debug_fetch
        st.session_state["run_key"] = None

    inputs = st.session_state.get("run_inputs")
    if not inputs:
        st.markdown(
            """
            <div style='text-align:center;padding:80px 0;'>
                <div style='font-size:3.5rem;margin-bottom:16px;'>📡</div>
                <div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem; color:#3d4f6e;letter-spacing:4px;'>
                    ENTER A TICKER AND PRESS RUN
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    ticker = sanitize_ticker(inputs["ticker"])
    if not ticker:
        st.error("Please enter a valid ticker symbol (letters, numbers, dot, dash).")
        return

    # Cloud mode overrides: skip LSTM, reduce history length, and enforce lightweight settings.
    cloud_env = is_cloud_env()
    effective_lightweight = inputs["lightweight"] or cloud_env
    effective_skip_lstm = inputs["skip_lstm"] or cloud_env

    # Stable run key prevents rerun storms on widget changes.
    run_key = make_run_key(ticker, inputs, effective_lightweight, effective_skip_lstm)
    st.session_state["run_key"] = run_key
    cached_run = inputs.get("use_cache", True) and run_key in st.session_state.get("results_by_key", {})

    states: Dict[int, str] = {}
    results: Dict[str, object] = {}

    def tick(idx: int, status: str) -> None:
        states[idx] = status
        render_pipeline(states, pipeline_ph)

    if cached_run:
        results = st.session_state.get("results_by_key", {}).get(run_key, {})
        states = {idx: "done" for idx in range(len(STEPS))}
        render_pipeline(states, pipeline_ph)
        st.info("✅ Using cached run results (no recomputation).")
    else:
        tick(0, "done")

    # Step 2: Stock data
    if not cached_run:
        tick(1, "active")
        status_ph = st.empty()
        status_ph.info("📡 **Step 2** — Fetching live stock data…")
        try:
            if inputs["use_cache"]:
                df, info = fetch_live_stock_cached(ticker)
            else:
                df, info = fetch_live_stock_uncached(ticker)
            results["df"] = df
            results["info"] = info
            tick(1, "done")
            status_ph.success(
                f"✅ **{info['name']}** — {len(df)} trading days loaded "
                f"({df.index[0].date()} → {df.index[-1].date()})"
            )
            if st.session_state.get("using_cached_data"):
                st.warning("⚠️ Using cached market data due to temporary Yahoo Finance limits.")
            if st.session_state.get("rate_limited"):
                st.warning("⏳ Yahoo Finance rate-limited requests. Please wait 1–2 minutes and retry.")
        except Exception as e:
            tick(1, "error")
            if st.session_state.get("rate_limited"):
                status_ph.error(
                    "Yahoo Finance temporarily rate-limited requests. Please wait 1–2 minutes and retry."
                )
            else:
                status_ph.error(f"❌ Stock data error: {e}")
            if st.session_state.get("debug_fetch") and st.session_state.get("fetch_logs"):
                with st.expander("Stock Fetch Debug Logs", expanded=False):
                    st.json(st.session_state.get("fetch_logs"))
            return

    df = results["df"]
    info = results["info"]
    articles = results.get("articles", [])
    cur = float(df["Close"].iloc[-1])
    currency = info["currency"]
    vol, vol_label = volatility(df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Price", fmt(cur, currency))
    c2.metric("Exchange", info["exchange"] or "—")
    c3.metric("Sector", info["sector"])
    c4.metric("Volatility", f"{vol}%", vol_label)
    mktcap = info["market_cap"]
    c5.metric("Market Cap", f"{mktcap/1e9:.1f}B {currency}" if mktcap else "—")

    st.markdown(
        f"<div class='live-badge'><span class='live-dot'></span>LIVE — data through {df.index[-1].date()}</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Step 3: News
    if not cached_run:
        tick(2, "active")
        news_ph = st.empty()
        if not news_api_key:
            news_ph.warning("⚠️ **Step 3** — NEWS_API_KEY missing. Using neutral sentiment.")
            tick(2, "done")
            results["articles"] = []
        else:
            news_ph.info(f"📰 **Step 3** — Fetching latest {inputs['max_articles']} news articles…")
            try:
                query = info["name"] if info["name"] and info["name"] != ticker else ticker
                if inputs["use_cache"]:
                    articles = fetch_live_news_cached(query, news_api_key, inputs["max_articles"])
                else:
                    articles = fetch_live_news_uncached(query, news_api_key, inputs["max_articles"])
                results["articles"] = articles
                tick(2, "done")
                news_ph.success(f"✅ **Step 3** — {len(articles)} live articles fetched")
            except Exception as e:
                tick(2, "error")
                news_ph.error(f"❌ News fetch error: {e}")
                results["articles"] = []

    # Step 4: Sentiment
    if not cached_run:
        tick(3, "active")
        sent_ph = st.empty()
        articles = results["articles"]

        if not articles:
            sent_ph.info("⚪ **Step 4** — No articles to classify. Using neutral sentiment.")
            tick(3, "done")
            results["sentiment"] = run_bert_sentiment([])
        else:
            sent_ph.info(f"🤖 **Step 4** — Running VADER sentiment on {len(articles)} headlines…")
            t0 = time.time()
            try:
                art_sig = articles_signature(articles)
                if inputs["use_cache"]:
                    sent = run_bert_sentiment_cached(articles, art_sig)
                else:
                    sent = run_bert_sentiment(articles)
                results["sentiment"] = sent
                tick(3, "done")
                sent_ph.success(
                    f"✅ **Step 4** — VADER done in {time.time()-t0:.1f}s  |  "
                    f"Positive {sent['pct_pos']}%  Neutral {sent['pct_neu']}%  Negative {sent['pct_neg']}%  |  "
                    f"Overall: **{sent['overall']}**"
                )
            except Exception as e:
                tick(3, "error")
                sent_ph.error(f"❌ Sentiment error: {e}")
                results["sentiment"] = run_bert_sentiment([])

    sent = results["sentiment"]
    with st.expander("📰 Sentiment Detail", expanded=False):
        s1, s2, s3 = st.columns(3)
        s1.metric("✅ Positive", f"{sent['pct_pos']}%", f"{sent['counts']['positive']} articles")
        s2.metric("⚪ Neutral", f"{sent['pct_neu']}%", f"{sent['counts']['neutral']} articles")
        s3.metric("🔴 Negative", f"{sent['pct_neg']}%", f"{sent['counts']['negative']} articles")
        st.markdown(
            f"""
            <div class='sent-bar-wrap'>
                <div style='width:{sent['pct_pos']}%;background:#3fb950'></div>
                <div style='width:{sent['pct_neu']}%;background:#d29922'></div>
                <div style='width:{sent['pct_neg']}%;background:#f85149'></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for art in sent["articles"]:
            css = {"positive": "news-pos", "negative": "news-neg", "neutral": "news-neu"}[art["sentiment"]]
            ico = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}[art["sentiment"]]
            st.markdown(
                f"""
                <div class='news-item {css}'>
                    {ico} <strong>{art['title']}</strong><br>
                    <span style='color:#6e7681;font-size:0.76rem;'>
                        {art['source']} · {art['date']} · {art['sentiment'].upper()} ({art['confidence']:.0%})
                        &nbsp;<a href='{art['url']}' target='_blank' style='color:#58a6ff;'>↗</a>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown(f"### 📈 Model Training & Prediction  (horizon: +{inputs['horizon']} days)")

    # Step 5: Features
    if not cached_run:
        tick(4, "active")
        feat_ph = st.empty()
        feat_ph.info("🔧 **Step 5** — Engineering features…")
        sig = data_signature(df)
        try:
            if inputs["use_cache"]:
                feat = engineer_features_cached(df, sig)
            else:
                feat = engineer_features_uncached(df)
            results["feat"] = feat
            results["sig"] = sig
            tick(4, "done")
            feat_ph.success(f"✅ **Step 5** — Features ready: {list(feat.columns)} ({len(feat)} rows)")
        except Exception as e:
            tick(4, "error")
            feat_ph.error(f"❌ Feature engineering error: {e}")
            return

    # Step 6: Linear Regression
    if not cached_run:
        tick(5, "active")
        lr_ph = st.empty()
        lr_ph.info("📐 **Step 6** — Training Linear Regression…")
        t0 = time.time()
        feat = results.get("feat")
        sig = results.get("sig")
        try:
            if inputs["use_cache"]:
                lr_res = train_linear_regression_cached(feat, inputs["horizon"], sig)
            else:
                lr_res = train_linear_regression_uncached(feat, inputs["horizon"])
            results["lr_res"] = lr_res
            tick(5, "done")
            lr_ph.success(
                f"✅ **Step 6** — Linear Regression in {time.time()-t0:.1f}s  |  "
                f"Predicted price: **{fmt(lr_res['predicted'], currency)}**  |  MAE: {lr_res['mae']}"
            )
        except Exception as e:
            tick(5, "error")
            lr_ph.error(f"❌ Linear Regression error: {e}")
            return

    # Step 7: Random Forest
    if not cached_run:
        tick(6, "active")
        rf_ph = st.empty()
        rf_ph.info("🌲 **Step 7** — Training Random Forest…")
        t0 = time.time()
        feat = results.get("feat")
        sig = results.get("sig")
        try:
            if inputs["use_cache"]:
                rf_res = train_random_forest_cached(feat, inputs["horizon"], sig)
            else:
                rf_res = train_random_forest_uncached(feat, inputs["horizon"])
            results["rf_res"] = rf_res
            tick(6, "done")
            rf_ph.success(
                f"✅ **Step 7** — Random Forest in {time.time()-t0:.1f}s  |  "
                f"Predicted price: **{fmt(rf_res['predicted'], currency)}**  |  MAE: {rf_res['mae']}"
            )
        except Exception as e:
            tick(6, "error")
            rf_ph.error(f"❌ Random Forest error: {e}")
            return

    # Step 8: LSTM
    if not cached_run:
        rf_res = results.get("rf_res")
        if effective_skip_lstm:
            tick(7, "done")
            lstm_res = {"model": "LSTM (skipped)", "predicted": rf_res["predicted"], "mae": rf_res["mae"]}
            results["lstm_res"] = lstm_res
            st.info("⏩ **Step 8** — LSTM skipped (toggle off in sidebar).")
        else:
            tick(7, "active")
            lstm_ph = st.empty()
            lstm_ph.info("🧠 **Step 8** — Training LSTM…")
            t0 = time.time()
            sig = results.get("sig")
            try:
                if inputs["use_cache"]:
                    lstm_res = train_lstm_cached(
                        df,
                        inputs["horizon"],
                        min(inputs["lstm_epochs"], 4 if cloud_env else inputs["lstm_epochs"]),
                        effective_lightweight,
                        sig,
                    )
                else:
                    lstm_res = train_lstm_uncached(
                        df,
                        inputs["horizon"],
                        min(inputs["lstm_epochs"], 4 if cloud_env else inputs["lstm_epochs"]),
                        effective_lightweight,
                    )
                results["lstm_res"] = lstm_res
                tick(7, "done")
                lstm_ph.success(
                    f"✅ **Step 8** — LSTM in {time.time()-t0:.1f}s  |  "
                    f"Predicted price: **{fmt(lstm_res['predicted'], currency)}**  |  MAE: {lstm_res['mae']}"
                )
            except Exception as e:
                tick(7, "error")
                lstm_ph.error(f"❌ LSTM error: {e}  |  Falling back to Random Forest.")
                lstm_res = {"model": "LSTM (fallback)", "predicted": rf_res["predicted"], "mae": rf_res["mae"]}
                results["lstm_res"] = lstm_res

    st.divider()

    # Step 9: Ensemble
    if not cached_run:
        tick(8, "active")
        ens_ph = st.empty()
        ens_ph.info("⚗️ **Step 9** — Computing MAE-weighted ensemble…")
        lr_res = results.get("lr_res")
        rf_res = results.get("rf_res")
        lstm_res = results.get("lstm_res")
        ens = ensemble(lr_res, rf_res, lstm_res, cur)
        results["ens"] = ens
        tick(8, "done")
        ens_ph.success(
            f"✅ **Step 9** — Ensemble prediction: **{fmt(ens['final'], currency)}**  |  "
            f"Change: {ens['pct_change']:+.2f}%  |  Confidence: {ens['confidence']}%"
        )

    st.markdown("#### Model Comparison")
    st.markdown(
        """
        <div class='model-row header'>
            <div>MODEL</div><div>PREDICTED PRICE</div><div>MAE (test set)</div><div>WEIGHT BASIS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for r in [lr_res, rf_res, lstm_res]:
        arrow = "🔼" if r["predicted"] >= cur else "🔽"
        st.markdown(
            f"""
            <div class='model-row'>
                <div style='color:#8b949e'>{r['model']}</div>
                <div style='color:#e6edf3'>{arrow} {fmt(r['predicted'], currency)}</div>
                <div style='color:#8b949e'>{r['mae']}</div>
                <div style='color:#3d4f6e'>1/MAE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Current Price", fmt(cur, currency))
    e2.metric(f"Ensemble (+{inputs['horizon']}d)", fmt(ens["final"], currency))
    delta_sign = "+" if ens["pct_change"] >= 0 else ""
    e3.metric("Expected Change", f"{delta_sign}{ens['pct_change']}%")
    e4.metric("Confidence", f"{ens['confidence']}%")

    st.markdown(
        f"""
        <div style='margin:10px 0'>
            <div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#3d4f6e;letter-spacing:2px;'>
                CONFIDENCE
            </div>
            <div class='conf-bar-wrap'>
                <div class='conf-bar-fill' style='width:{ens['confidence']}%'></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📉 Historical Price (Last 1Y)", expanded=True):
        chart_points = 126 if cloud_env else 252
        price_view = df[["Close"]].tail(chart_points)
        st.line_chart(price_view, use_container_width=True, color="#1f6feb")

    st.divider()

    # Step 10: Signal
    if not cached_run:
        tick(9, "active")
        sig_ph = st.empty()
        sig_ph.info("🤖 **Step 10** — Generating final signal…")
        decision = make_signal(ens, sent)
        sig = decision["signal"]
        results["decision"] = decision
        tick(9, "done")
        sig_ph.success(f"✅ **Step 10** — Signal: **{sig}**")
    else:
        decision = results.get("decision")
        ens = results.get("ens")
        lr_res = results.get("lr_res")
        rf_res = results.get("rf_res")
        lstm_res = results.get("lstm_res")
        sent = results.get("sentiment")
        sig = decision["signal"]

    st.markdown("### 🎯 Final AI Signal")
    st.markdown(
        f"""
        <div class='signal-box signal-{sig}'>
            <div class='signal-word-{sig}'>{sig}</div>
            <div class='signal-reason'>{decision['reason']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📋 Full Run Summary")
    summary_data = {
        "Ticker": ticker,
        "Company": info["name"],
        "Data Range": f"{df.index[0].date()} → {df.index[-1].date()}",
        "Current Price": fmt(cur, currency),
        "Linear Regression Pred": fmt(lr_res["predicted"], currency),
        "Random Forest Pred": fmt(rf_res["predicted"], currency),
        "LSTM Pred": fmt(lstm_res["predicted"], currency),
        f"Ensemble Pred (+{inputs['horizon']}d)": fmt(ens["final"], currency),
        "Expected Growth": f"{ens['pct_change']:+.2f}%",
        "Model Confidence": f"{ens['confidence']}%",
        "Annualised Volatility": f"{vol}% ({vol_label})",
        "Sentiment Score": f"{sent['score']:.2f} / 1.00  ({sent['overall']})",
        "Articles Analysed": str(len(articles)),
        "🤖 AI Signal": sig,
        "Reason": decision["reason"],
        "Run Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.table(pd.DataFrame(summary_data.items(), columns=["Metric", "Value"]).set_index("Metric"))

    cloud_state = "Cloud" if is_cloud_env() else "Local"
    st.caption(
        f"Runtime: {cloud_state} | Cache TTL: {DEFAULT_CACHE_TTL}s | Lightweight: {effective_lightweight}"
    )
    st.caption(
        "⚠️ **Disclaimer**: Educational / research use only. Not financial advice. "
        "Past performance does not guarantee future results."
    )

    if not cached_run:
        results.update(
            {
                "cur": cur,
                "currency": currency,
                "vol": vol,
                "vol_label": vol_label,
            }
        )
        if inputs.get("use_cache", True):
            st.session_state.setdefault("results_by_key", {})[run_key] = results


def fetch_backtest_data(ticker: str, period: str = "2y") -> Tuple[pd.DataFrame, Dict]:
    """Fetch historical data for backtesting using the same logic as app.py."""
    session = get_yf_session()
    limiter = RateLimiter(YF_MIN_INTERVAL)

    candidates = build_ticker_candidates(ticker)
    df = pd.DataFrame()
    ok = False

    for cand in candidates:
        try:
            limiter.wait()
            try:
                df = yf.download(
                    cand,
                    progress=False,
                    auto_adjust=False,
                    group_by="column",
                    threads=False,
                    timeout=YF_TIMEOUT,
                    session=session,
                    period=period,
                    interval="1d",
                )
            except TypeError:
                df = yf.download(
                    cand,
                    progress=False,
                    auto_adjust=False,
                    period=period,
                    interval="1d",
                )
            ok, df, reason = validate_ohlcv(df)
            if ok:
                ticker = cand
                break
        except Exception:
            continue

    if not ok:
        raise ValueError(f"No price data found for '{ticker}'")

    try:
        info = get_ticker_info_cached(ticker)
    except Exception:
        info = {"name": ticker, "currency": "USD", "exchange": "", "sector": "N/A", "market_cap": 0}

    return df, info


def run_backtest(ticker: str, horizon: int = 7, n_windows: int = 5,
                 train_window: int = 252, test_window: int = 30) -> Dict:
    """Rolling window backtest using the exact same models as app.py."""
    print(f"\n{'='*70}")
    print(f"BACKTEST: {ticker} | Horizon: {horizon}d | Windows: {n_windows}")
    print(f"{'='*70}")

    df, info = fetch_backtest_data(ticker, period="2y")
    print(f"Loaded {len(df)} rows | {df.index[0].date()} -> {df.index[-1].date()}")

    min_required = train_window + test_window + horizon
    if len(df) < min_required:
        raise ValueError(f"Need {min_required} rows, got {len(df)}")

    results = {"ticker": ticker, "horizon": horizon, "windows": []}

    for w in range(n_windows):
        end_idx = len(df) - w * test_window
        start_idx = end_idx - train_window - test_window
        if start_idx < 0:
            break

        train_df = df.iloc[start_idx:start_idx + train_window].copy()
        test_df = df.iloc[start_idx + train_window:end_idx].copy()

        current_price = float(train_df["Close"].iloc[-1])

        # Feature engineering
        train_feat = engineer_features_uncached(train_df)

        # Train models
        lr = train_linear_regression_uncached(train_feat, horizon)
        rf = train_random_forest_uncached(train_feat, horizon)
        lstm = train_lstm_uncached(train_df, horizon, epochs=12, lightweight=True)

        # Ensemble
        ens = ensemble(lr, rf, lstm, current_price)

        # Actual future price
        actual_idx = horizon - 1
        actual = None
        actual_pct = None
        direction_correct = None
        ens_error = None
        ens_error_pct = None

        actual_date = None
        if actual_idx < len(test_df):
            actual = float(test_df["Close"].iloc[actual_idx])
            actual_date = test_df.index[actual_idx].date()
            actual_pct = round((actual - current_price) / current_price * 100, 2)
            direction_correct = (ens["pct_change"] > 0) == (actual_pct > 0)
            ens_error = round(abs(ens["final"] - actual), 2)
            ens_error_pct = round(ens_error / actual * 100, 2)

        window = {
            "window": w + 1,
            "train_start": train_df.index[0].date(),
            "train_end": train_df.index[-1].date(),
            "test_start": test_df.index[0].date(),
            "actual_date": actual_date,
            "current": round(current_price, 2),
            "actual": round(actual, 2) if actual else None,
            "actual_pct": actual_pct,
            "lr_pred": lr["predicted"],
            "rf_pred": rf["predicted"],
            "lstm_pred": lstm["predicted"],
            "ensemble_pred": ens["final"],
            "ensemble_pct": ens["pct_change"],
            "ensemble_error": ens_error,
            "ensemble_error_pct": ens_error_pct,
            "direction_correct": direction_correct,
            "confidence": ens["confidence"],
            "lr_mae": lr["mae"],
            "rf_mae": rf["mae"],
            "lstm_mae": lstm["mae"],
        }
        results["windows"].append(window)

        if actual is not None:
            print(
                f"  Window {w+1}: Current ${current_price:.2f} | "
                f"Actual (+{horizon}d) ${actual:.2f} ({actual_pct:+.2f}%) | "
                f"Ens ${ens['final']:.2f} ({ens['pct_change']:+.2f}%) | "
                f"Err {ens_error_pct:.2f}% | Dir {'OK' if direction_correct else 'FAIL'}"
            )
        else:
            print(f"  Window {w+1}: skipped (not enough test data)")

    # Aggregate metrics
    valid = [w for w in results["windows"] if w["actual"] is not None]
    if valid:
        errors = [w["ensemble_error"] for w in valid]
        error_pcts = [w["ensemble_error_pct"] for w in valid]
        dirs = [w["direction_correct"] for w in valid]

        results["aggregate"] = {
            "n_windows": len(valid),
            "mean_abs_error": round(float(np.mean(errors)), 2),
            "median_abs_error": round(float(np.median(errors)), 2),
            "mean_error_pct": round(float(np.mean(error_pcts)), 2),
            "median_error_pct": round(float(np.median(error_pcts)), 2),
            "directional_accuracy": round(float(np.mean(dirs)) * 100, 1),
            "max_error": round(float(np.max(errors)), 2),
            "min_error": round(float(np.min(errors)), 2),
        }

        print(f"\n{'='*70}")
        print("AGGREGATE RESULTS")
        print(f"{'='*70}")
        for k, v in results["aggregate"].items():
            print(f"  {k}: {v}")

    return results


def _results_to_rows(results: Dict) -> List[Dict]:
    """Convert a single-ticker backtest result dict into per-window rows."""
    rows = []
    ticker = results.get("ticker")
    horizon = results.get("horizon")
    for w in results.get("windows", []):
        rows.append({
            "ticker": ticker,
            "horizon": horizon,
            "window": w["window"],
            "train_start": w["train_start"],
            "train_end": w["train_end"],
            "prediction_date": w["train_end"],
            "actual_date": w["actual_date"],
            "current_price": w["current"],
            "actual_price": w["actual"],
            "ensemble_predicted": w["ensemble_pred"],
            "lr_predicted": w["lr_pred"],
            "rf_predicted": w["rf_pred"],
            "lstm_predicted": w["lstm_pred"],
            "actual_pct_change": w["actual_pct"],
            "ensemble_pct_change": w["ensemble_pct"],
            "error_dollars": w["ensemble_error"],
            "error_percent": w["ensemble_error_pct"],
            "direction_correct": w["direction_correct"],
            "confidence": w["confidence"],
        })
    return rows


def save_backtest_results(results: Dict, path: str = "Backtracking/backtest_results.csv") -> None:
    """Save per-window backtest results to CSV.

    If the requested file is open in Excel/another program, a timestamped
    copy is written instead.
    """
    rows = _results_to_rows(results)
    if not rows:
        return

    df = pd.DataFrame(rows)
    written_path = path
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        written_path = path.replace(".csv", f"_{ts}.csv")
        df.to_csv(written_path, index=False)
        print(f"Warning: {path} is locked. Saved to {written_path} instead.")

    print(f"\nResults saved to {written_path}")
    return written_path


def parse_backtest_cli_args(args: List[str]) -> Tuple[List[str], int, int]:
    """Parse CLI args for backtest entry points.

    Ticker arguments may be comma-separated, space-separated, or quoted
    individually. The first pure-integer argument is treated as the forecast
    horizon; the second pure-integer argument is treated as the number of
    rolling windows.

    Returns
    -------
    (tickers, horizon, n_windows)
    """
    tickers: List[str] = []
    horizon = 7
    n_windows = 5
    seen_horizon = False

    for raw in args:
        cleaned = raw.strip()
        if not cleaned:
            continue
        parts = [
            p.strip().strip('"').strip("'").upper()
            for p in cleaned.replace(",", " ").split()
            if p.strip()
        ]
        for p in parts:
            if p.isdigit():
                val = int(p)
                if not seen_horizon:
                    horizon = val
                    seen_horizon = True
                else:
                    n_windows = val
            else:
                tickers.append(p)

    if not tickers:
        tickers = ["AAPL"]
    return tickers, horizon, n_windows


def run_cli_backtest():
    """Command-line entry point for backtesting.

    Usage:
        python backtest.py <ticker> <horizon> <n_windows>
        python backtest.py <ticker1,ticker2,...> <horizon> <n_windows>

    Examples:
        python backtest.py TSLA 7 10
        python backtest.py AAPL,MSFT,TSLA 7 5
        python backtest.py "HDFCBANK.NS" "TCS.NS" 7 5
    """
    tickers, horizon, n_windows = parse_backtest_cli_args(sys.argv[1:])

    # Ensure no stale Streamlit caches influence the backtest.
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        print("Streamlit caches cleared.")
    except Exception:
        pass

    out_path = "Backtracking/backtest_results.csv"

    if len(tickers) == 1:
        results = run_backtest(tickers[0], horizon=horizon, n_windows=n_windows)
        save_backtest_results(results, out_path)
    else:
        print(f"Running comprehensive backtest for {', '.join(tickers)} | horizon={horizon} | windows={n_windows}")
        detailed = run_comprehensive_backtest_detailed(tickers, horizons=[horizon], n_windows=n_windows)
        summary = run_comprehensive_backtest(detailed_df=detailed)

        if not summary.empty:
            print("\n" + "=" * 60)
            print("SUMMARY TABLE")
            print("=" * 60)
            print(summary.to_string(index=False))

        if not detailed.empty:
            try:
                detailed.to_csv(out_path, index=False)
                print(f"\nDetailed results ({len(detailed)} rows) saved to: {out_path}")
            except PermissionError:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                alt_path = out_path.replace(".csv", f"_{ts}.csv")
                detailed.to_csv(alt_path, index=False)
                print(f"\nWarning: {out_path} is locked. Detailed results saved to {alt_path}")


def evaluate_prediction_accuracy(ticker: str, horizon: int = 7) -> Dict:
    """Evaluate recent prediction accuracy for a single ticker.

    Runs a short rolling-window backtest and returns aggregate accuracy
    metrics. This is a convenience wrapper around :func:`run_backtest`.
    """
    results = run_backtest(ticker, horizon=horizon, n_windows=3)
    agg = results.get("aggregate", {})
    return {
        "ticker": ticker,
        "horizon": horizon,
        "directional_accuracy": agg.get("directional_accuracy"),
        "mean_error_pct": agg.get("mean_error_pct"),
        "mean_abs_error": agg.get("mean_abs_error"),
        "n_windows": agg.get("n_windows", 0),
    }


def run_comprehensive_backtest_detailed(
    tickers: List[str],
    horizons: List[int] = None,
    n_windows: int = 5,
) -> pd.DataFrame:
    """Run rolling-window backtests across multiple tickers and horizons.

    Returns a per-window detailed DataFrame suitable for CSV export.
    Columns match those produced by :func:`save_backtest_results`.

    Parameters
    ----------
    tickers: list of str
        Stock symbols to backtest.
    horizons: list of int, optional
        Forecast horizons in days. Defaults to [7].
    n_windows: int
        Number of rolling windows per (ticker, horizon).

    Returns
    -------
    pd.DataFrame
        One row per backtest window across all tickers/horizons.
    """
    if horizons is None:
        horizons = [7]

    all_rows = []
    for ticker in tickers:
        for horizon in horizons:
            try:
                results = run_backtest(ticker, horizon=horizon, n_windows=n_windows)
                all_rows.extend(_results_to_rows(results))
            except Exception as e:
                print(f"  Skipping {ticker} @ {horizon}d: {e}")

    return pd.DataFrame(all_rows)


def run_comprehensive_backtest(
    tickers: List[str] = None,
    horizons: List[int] = None,
    n_windows: int = 5,
    detailed_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Run rolling-window backtests across multiple tickers and horizons.

    Parameters
    ----------
    tickers: list of str, optional
        Stock symbols to backtest. Required unless `detailed_df` is provided.
    horizons: list of int, optional
        Forecast horizons in days. Defaults to [7].
    n_windows: int
        Number of rolling windows per (ticker, horizon).
    detailed_df: pd.DataFrame, optional
        Pre-computed detailed results from :func:`run_comprehensive_backtest_detailed`.
        If provided, aggregation is performed on this DataFrame instead of
        re-running the backtests.

    Returns
    -------
    pd.DataFrame
        One row per (ticker, horizon) with aggregate metrics.
    """
    if detailed_df is None:
        if tickers is None:
            raise ValueError("Either tickers or detailed_df must be provided.")
        detailed = run_comprehensive_backtest_detailed(tickers, horizons=horizons, n_windows=n_windows)
    else:
        detailed = detailed_df

    if detailed.empty:
        return pd.DataFrame(columns=[
            "ticker", "horizon", "n_windows", "mean_abs_error",
            "median_abs_error", "mean_error_pct", "directional_accuracy",
            "max_error", "min_error",
        ])

    summary_rows = []
    for (ticker, horizon), group in detailed.groupby(["ticker", "horizon"]):
        valid = group[group["actual_price"].notna()]
        if valid.empty:
            summary_rows.append({
                "ticker": ticker,
                "horizon": horizon,
                "n_windows": 0,
                "mean_abs_error": None,
                "median_abs_error": None,
                "mean_error_pct": None,
                "directional_accuracy": None,
                "max_error": None,
                "min_error": None,
            })
            continue

        errors = valid["error_dollars"].astype(float)
        error_pcts = valid["error_percent"].astype(float)
        dirs = valid["direction_correct"].astype(bool)

        summary_rows.append({
            "ticker": ticker,
            "horizon": horizon,
            "n_windows": len(valid),
            "mean_abs_error": round(float(errors.mean()), 2),
            "median_abs_error": round(float(errors.median()), 2),
            "mean_error_pct": round(float(error_pcts.mean()), 2),
            "directional_accuracy": round(float(dirs.mean()) * 100, 1),
            "max_error": round(float(errors.max()), 2),
            "min_error": round(float(errors.min()), 2),
        })

    return pd.DataFrame(summary_rows)

if __name__ == "__main__":
    if IN_STREAMLIT:
        main()
    else:
        run_cli_backtest()