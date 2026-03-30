"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              AI STOCK ANALYST  —  STRICT LIVE SEQUENTIAL PIPELINE            ║
║                                                                              ║
║  WORKFLOW (triggered on every run, no caching of results):                   ║
║  Step 1 → User enters ticker                                                 ║
║  Step 2 → Fetch LIVE stock data (yfinance, up to today)                      ║
║  Step 3 → Fetch LIVE news (NewsAPI, today's articles)                        ║
║  Step 4 → Run BERT sentiment on live headlines                               ║
║  Step 5 → Engineer features from live data                                   ║
║  Step 6 → Train Linear Regression on live data  → predict                    ║
║  Step 7 → Train Random Forest on live data      → predict                    ║
║  Step 8 → Train LSTM on live data               → predict                    ║
║  Step 9 → Ensemble all 3 predictions                                         ║
║  Step 10 → Combine sentiment + ensemble → BUY / SELL / HOLD                  ║
║                                                                              ║
║  Run:  streamlit run app.py                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Stdlib ─────────────────────────────────────────────────────────────────
import warnings, logging, time
from datetime import datetime

warnings.filterwarnings("ignore")
os_env_set = __import__("os").environ
os_env_set["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)

# ── Third-party ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from transformers import pipeline as hf_pipeline

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Stock Analyst",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS  — Bloomberg-terminal meets brutalist design
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

*, html, body { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1,h2,h3,h4, .mono { font-family: 'IBM Plex Mono', monospace !important; }

.stApp { background: #080b10; color: #c9d1d9; }
.main .block-container { padding-top: 1.5rem; max-width: 1300px; }

/* ── HEADER ── */
.app-header {
    border-bottom: 1px solid #1c2333;
    padding-bottom: 1.2rem;
    margin-bottom: 1.5rem;
}
.app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.app-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #3d4f6e;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── WORKFLOW STEPS ── */
.step-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    border-radius: 6px;
    margin: 5px 0;
    background: #0d1117;
    border: 1px solid #1c2333;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    transition: all 0.3s;
}
.step-row.active {
    background: #0d1e33;
    border-color: #1f6feb;
    color: #58a6ff;
}
.step-row.done {
    background: #0d1f17;
    border-color: #196c3b;
    color: #3fb950;
}
.step-row.error {
    background: #1f0d0d;
    border-color: #6e1313;
    color: #f85149;
}
.step-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-wait  { background: #3d4f6e; }
.dot-active{ background: #1f6feb; box-shadow: 0 0 8px #1f6feb; animation: blink 1s infinite; }
.dot-done  { background: #3fb950; }
.dot-error { background: #f85149; }

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── SIGNAL ── */
.signal-box {
    border-radius: 10px;
    padding: 32px 20px;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
}
.signal-BUY  { background:#051a0f; border:2px solid #3fb950; }
.signal-SELL { background:#1a0505; border:2px solid #f85149; }
.signal-HOLD { background:#111108; border:2px solid #d29922; }
.signal-word-BUY  { font-size:3.6rem; font-weight:700; color:#3fb950; letter-spacing:10px; }
.signal-word-SELL { font-size:3.6rem; font-weight:700; color:#f85149; letter-spacing:10px; }
.signal-word-HOLD { font-size:3.6rem; font-weight:700; color:#d29922; letter-spacing:10px; }
.signal-reason { color:#8b949e; margin-top:10px; font-size:0.85rem; }

/* ── METRIC CARDS ── */
.metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin:12px 0; }
.mcard {
    background:#0d1117;
    border:1px solid #1c2333;
    border-radius:8px;
    padding:14px 16px;
}
.mcard-label { font-size:0.68rem; color:#3d4f6e; letter-spacing:2px; text-transform:uppercase; font-family:'IBM Plex Mono',monospace; }
.mcard-value { font-size:1.25rem; font-weight:600; color:#e6edf3; font-family:'IBM Plex Mono',monospace; margin-top:4px; }
.mcard-sub   { font-size:0.72rem; color:#8b949e; margin-top:2px; }

/* ── NEWS ── */
.news-item {
    border-left: 3px solid #1c2333;
    padding: 8px 14px;
    margin: 6px 0;
    font-size: 0.83rem;
    background: #0d1117;
    border-radius: 0 6px 6px 0;
}
.news-pos { border-left-color: #3fb950; }
.news-neg { border-left-color: #f85149; }
.news-neu { border-left-color: #d29922; }

/* ── MODEL TABLE ── */
.model-row {
    display:grid;
    grid-template-columns: 160px 1fr 1fr 1fr;
    gap:8px;
    padding:10px 14px;
    border-radius:6px;
    margin:4px 0;
    background:#0d1117;
    border:1px solid #1c2333;
    font-family:'IBM Plex Mono',monospace;
    font-size:0.82rem;
    align-items:center;
}
.model-row.header { color:#3d4f6e; font-size:0.7rem; letter-spacing:2px; background:transparent; border-color:transparent; }

/* ── CONFIDENCE BAR ── */
.conf-bar-wrap { background:#1c2333; border-radius:4px; height:10px; overflow:hidden; margin-top:6px; }
.conf-bar-fill { height:100%; border-radius:4px; background:linear-gradient(90deg,#1f6feb,#58a6ff); }

/* ── SENTIMENT BAR ── */
.sent-bar-wrap { display:flex; height:14px; border-radius:6px; overflow:hidden; margin:8px 0; }

/* ── LIVE BADGE ── */
.live-badge {
    display:inline-block;
    background:#1a1f2e;
    border:1px solid #1f6feb;
    color:#58a6ff;
    font-family:'IBM Plex Mono',monospace;
    font-size:0.65rem;
    letter-spacing:3px;
    padding:2px 10px;
    border-radius:20px;
}
.live-dot {
    display:inline-block;
    width:6px;height:6px;
    border-radius:50%;
    background:#3fb950;
    margin-right:5px;
    animation:blink 1.4s infinite;
    vertical-align:middle;
}

/* ── STREAMLIT OVERRIDES ── */
.stButton > button {
    background: #1f6feb;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 2px;
    padding: 0.55rem 1.8rem;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover { background: #388bfd; }
.stTextInput > div > div > input {
    background: #0d1117;
    border: 1px solid #1c2333;
    border-radius: 6px;
    color: #e6edf3;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;
}
div[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #1c2333;
    border-radius: 8px;
    padding: 12px;
}
.stSidebar { background: #0d1117; }
.stExpander { background: #0d1117; border: 1px solid #1c2333; border-radius:8px; }
section[data-testid="stSidebar"] { background: #0d1117; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP TRACKER  — renders the live pipeline progress in the sidebar
# ══════════════════════════════════════════════════════════════════════════════

STEPS = [
    ("01", "User Input"),
    ("02", "Fetch Live Stock Data"),
    ("03", "Fetch Live News"),
    ("04", "BERT Sentiment Analysis"),
    ("05", "Feature Engineering"),
    ("06", "Train Linear Regression"),
    ("07", "Train Random Forest"),
    ("08", "Train LSTM"),
    ("09", "Ensemble Predictions"),
    ("10", "Generate Signal"),
]

def render_pipeline(states: dict, placeholder):
    """Render the step pipeline into a given sidebar placeholder."""
    html = "<div style='margin-top:8px;'>"
    for idx, (num, label) in enumerate(STEPS):
        state = states.get(idx, "wait")
        dot_cls  = {"wait":"dot-wait","active":"dot-active","done":"dot-done","error":"dot-error"}[state]
        row_cls  = {"wait":"","active":" active","done":" done","error":" error"}[state]
        icon     = {"wait":"○","active":"◉","done":"✓","error":"✗"}[state]
        html += f"""
        <div class='step-row{row_cls}'>
            <span class='step-dot {dot_cls}'></span>
            <span style='color:#3d4f6e;'>STEP {num}</span>
            <span style='flex:1'>{label}</span>
            <span>{icon}</span>
        </div>"""
    html += "</div>"
    placeholder.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS  (all operate on fresh live data, no result caching)
# ══════════════════════════════════════════════════════════════════════════════

# ── STEP 2: Live Stock Data ────────────────────────────────────────────────
def fetch_live_stock(ticker: str) -> tuple[pd.DataFrame, dict]:
    """
    Download up-to-today OHLCV via yfinance.
    Returns (dataframe, info_dict). Raises ValueError on failure.
    """
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data found for '{ticker}'. Check the ticker symbol.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close","Volume"]].dropna()

    try:
        raw = yf.Ticker(ticker).info
        info = {
            "name":       raw.get("longName") or raw.get("shortName") or ticker,
            "currency":   raw.get("currency", "USD"),
            "exchange":   raw.get("exchange", ""),
            "sector":     raw.get("sector", "N/A"),
            "market_cap": raw.get("marketCap", 0),
        }
    except Exception:
        info = {"name": ticker, "currency": "USD", "exchange": "", "sector": "N/A", "market_cap": 0}

    return df, info


# ── STEP 3: Live News ──────────────────────────────────────────────────────
def fetch_live_news(ticker: str, company: str, api_key: str, n: int = 15) -> list[dict]:
    """
    Fetch today/recent news from NewsAPI.
    Query uses company name first; falls back to ticker root.
    """
    query = company if company and company != ticker else ticker.replace(".NS","").replace(".BO","")
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={requests.utils.quote(query)}"
        "&sortBy=publishedAt&language=en"
        f"&pageSize={n}&apiKey={api_key}"
    )
    resp = requests.get(url, timeout=12)
    if resp.status_code != 200:
        raise ConnectionError(f"NewsAPI error {resp.status_code}: {resp.json().get('message','')}")
    articles = resp.json().get("articles", [])
    return [
        {
            "title":     a["title"],
            "desc":      a.get("description") or "",
            "url":       a.get("url",""),
            "date":      (a.get("publishedAt") or "")[:10],
            "source":    a.get("source",{}).get("name",""),
        }
        for a in articles if a.get("title")
    ]


# ── STEP 4: BERT Sentiment ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)   # model weights cached once; inference always live
def _load_bert():
    return hf_pipeline(
        "text-classification",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        top_k=1,
    )

def run_bert_sentiment(articles: list[dict]) -> dict:
    """
    Classify each article with BERT.
    Stars 1-2 → negative, 3 → neutral, 4-5 → positive.
    """
    model = _load_bert()
    scored = []
    for art in articles:
        text = (art["title"] + ". " + art["desc"])[:512]
        try:
            out       = model(text)[0][0]
            stars     = int(out["label"][0])
            sentiment = "positive" if stars >= 4 else ("neutral" if stars == 3 else "negative")
            conf      = round(out["score"], 4)
        except Exception:
            sentiment, conf = "neutral", 0.5
        scored.append({**art, "sentiment": sentiment, "confidence": conf})

    total = len(scored) or 1
    cnts  = {k: sum(1 for s in scored if s["sentiment"]==k) for k in ("positive","neutral","negative")}
    score = (cnts["positive"] * 1.0 + cnts["neutral"] * 0.5) / total   # 0–1

    return {
        "articles":        scored,
        "counts":          cnts,
        "pct_pos":         round(cnts["positive"] / total * 100, 1),
        "pct_neu":         round(cnts["neutral"]  / total * 100, 1),
        "pct_neg":         round(cnts["negative"] / total * 100, 1),
        "score":           round(score, 4),
        "overall":         "Positive" if score > 0.6 else ("Negative" if score < 0.4 else "Neutral"),
    }


# ── STEP 5: Feature Engineering ───────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive technical indicators from live OHLCV; drop NaN rows."""
    d = df[["Close"]].copy()
    d["MA7"]   = d["Close"].rolling(7).mean()
    d["MA21"]  = d["Close"].rolling(21).mean()
    d["MA50"]  = d["Close"].rolling(50).mean()
    d["Std7"]  = d["Close"].rolling(7).std()
    d["Pct1"]  = d["Close"].pct_change(1)
    d["Pct5"]  = d["Close"].pct_change(5)
    d["Pct20"] = d["Close"].pct_change(20)
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"]   = 100 - 100 / (1 + gain / (loss + 1e-9))
    return d.dropna()


# ── STEP 6: Linear Regression ─────────────────────────────────────────────
def train_linear_regression(feat: pd.DataFrame, horizon: int) -> dict:
    """Train LR on live feature set; return future price + MAE."""
    X = feat.copy()
    y = X["Close"].shift(-horizon).dropna()
    X = X.iloc[:len(y)]

    split = int(len(X) * 0.85)
    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(X.iloc[:split]);  ytr = y.iloc[:split]
    Xte = scaler.transform(X.iloc[split:]);       yte = y.iloc[split:]

    model = LinearRegression().fit(Xtr, ytr)
    mae   = round(mean_absolute_error(yte, model.predict(Xte)), 4)
    pred  = float(model.predict(scaler.transform(feat.iloc[[-1]]))[0])
    return {"model": "Linear Regression", "predicted": round(pred, 2), "mae": mae}


# ── STEP 7: Random Forest ─────────────────────────────────────────────────
def train_random_forest(feat: pd.DataFrame, horizon: int) -> dict:
    """Train RF on live feature set; return future price + MAE."""
    X = feat.copy()
    y = X["Close"].shift(-horizon).dropna()
    X = X.iloc[:len(y)]

    split = int(len(X) * 0.85)
    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(X.iloc[:split]);  ytr = y.iloc[:split]
    Xte = scaler.transform(X.iloc[split:]);       yte = y.iloc[split:]

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    mae  = round(mean_absolute_error(yte, model.predict(Xte)), 4)
    pred = float(model.predict(scaler.transform(feat.iloc[[-1]]))[0])
    return {"model": "Random Forest", "predicted": round(pred, 2), "mae": mae}


# ── STEP 8: LSTM ──────────────────────────────────────────────────────────
def train_lstm(df: pd.DataFrame, horizon: int, epochs: int = 8) -> dict:
    """
    Train a 2-layer LSTM on live closing prices.
    Uses iterative multi-step prediction for horizon days.
    """
    prices = df["Close"].values.astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    LOOK_BACK = 30
    # Build sequences
    Xs, ys = [], []
    for i in range(LOOK_BACK, len(scaled) - horizon):
        Xs.append(scaled[i - LOOK_BACK:i])
        ys.append(scaled[i + horizon - 1])   # predict price `horizon` steps ahead
    Xs = np.array(Xs).reshape(-1, LOOK_BACK, 1)
    ys = np.array(ys)

    split = int(len(Xs) * 0.85)
    Xtr, Xte = Xs[:split], Xs[split:]
    ytr, yte  = ys[:split], ys[split:]

    model = Sequential([
        LSTM(48, return_sequences=True, input_shape=(LOOK_BACK, 1)),
        Dropout(0.2),
        LSTM(24),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="huber")  # Huber = robust to outliers

    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0)
    model.fit(
        Xtr, ytr,
        validation_split=0.1,
        epochs=epochs,
        batch_size=32,
        callbacks=[es],
        verbose=0,
    )

    test_pred = scaler.inverse_transform(model.predict(Xte, verbose=0))
    test_true = scaler.inverse_transform(yte.reshape(-1, 1))
    mae = round(float(mean_absolute_error(test_true, test_pred)), 4)

    # Iterative forecast: feed predictions back as input
    seq = scaled[-LOOK_BACK:].copy()
    for _ in range(horizon):
        inp  = seq[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
        nxt  = model.predict(inp, verbose=0)[0][0]
        seq  = np.append(seq, nxt)
    pred = float(scaler.inverse_transform([[seq[-1]]])[0][0])

    return {"model": "LSTM", "predicted": round(pred, 2), "mae": mae}


# ── STEP 9: Ensemble ──────────────────────────────────────────────────────
def ensemble(lr: dict, rf: dict, lstm: dict, current: float) -> dict:
    """
    MAE-inverse weighted average of the 3 model predictions.
    Confidence = 1 − (std / mean) of predictions, scaled 0-100.
    """
    def inv(x): return 1.0 / (x + 1e-6)
    wlr, wrf, wlstm = inv(lr["mae"]), inv(rf["mae"]), inv(lstm["mae"])
    total_w  = wlr + wrf + wlstm
    final    = (wlr * lr["predicted"] + wrf * rf["predicted"] + wlstm * lstm["predicted"]) / total_w

    preds      = np.array([lr["predicted"], rf["predicted"], lstm["predicted"]])
    cv         = preds.std() / (preds.mean() + 1e-9)
    confidence = round(max(0.0, min(1.0, 1 - cv)) * 100, 1)
    pct        = round((final - current) / current * 100, 2)

    return {
        "final":      round(final, 2),
        "pct_change": pct,
        "confidence": confidence,
        "current":    round(current, 2),
    }


# ── STEP 10: Decision ─────────────────────────────────────────────────────
def make_signal(ens: dict, sent: dict) -> dict:
    """
    BUY  → sentiment > 0.55 AND predicted growth > +1%
    SELL → sentiment < 0.45 AND predicted growth < -1%
    HOLD → everything else
    """
    ss  = sent["score"]
    g   = ens["pct_change"]
    if   ss > 0.55 and g >  1.0: signal = "BUY"
    elif ss < 0.45 and g < -1.0: signal = "SELL"
    else:                         signal = "HOLD"

    reasons = {
        "BUY":  f"Bullish sentiment ({ss:.0%}) combined with +{g:.1f}% projected growth",
        "SELL": f"Bearish sentiment ({ss:.0%}) combined with {g:.1f}% projected decline",
        "HOLD": f"Mixed signals — sentiment {ss:.0%}, projected change {g:+.1f}%",
    }
    return {"signal": signal, "reason": reasons[signal]}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def fmt(val: float, currency: str = "USD") -> str:
    sym = {"USD":"$","INR":"₹","EUR":"€","GBP":"£"}.get(currency, currency+" ")
    return f"{sym}{val:,.2f}"

def volatility(df: pd.DataFrame) -> tuple[float, str]:
    v = df["Close"].pct_change().std() * (252**0.5) * 100
    lbl = "Low 🟢" if v < 20 else ("Medium 🟡" if v < 40 else "High 🔴")
    return round(v, 1), lbl


# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class='app-header'>
        <div class='app-title'>📡 AI STOCK ANALYST</div>
        <div class='app-sub'>LIVE DATA · BERT · LINEAR REGRESSION · RANDOM FOREST · LSTM · ENSEMBLE</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#3d4f6e;letter-spacing:3px;'>CONFIGURATION</div>", unsafe_allow_html=True)
        st.markdown("---")
        news_api_key  = st.text_input("NewsAPI Key", type="password", placeholder="Paste key from newsapi.org")
        horizon       = st.slider("Forecast Horizon (days)", 5, 30, 7)
        max_articles  = st.slider("News Articles to Fetch", 5, 20, 12)
        lstm_epochs   = st.slider("LSTM Epochs", 5, 20, 8)
        skip_lstm     = st.toggle("Skip LSTM (faster)", value=False)

        st.markdown("---")
        st.markdown("<div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#3d4f6e;letter-spacing:3px;'>QUICK TICKERS</div>", unsafe_allow_html=True)
        quick = ["AAPL","TSLA","TCS.NS","RELIANCE.NS","INFY.NS","MSFT","NVDA","HDFCBANK.NS"]
        cols  = st.columns(2)
        for i, t in enumerate(quick):
            if cols[i%2].button(t, key=f"q_{t}"):
                st.session_state["ticker"] = t

        st.markdown("---")
        st.markdown("<div style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#3d4f6e;'>PIPELINE PROGRESS</div>", unsafe_allow_html=True)
        pipeline_ph = st.empty()   # placeholder for live step tracker

    # Render initial (all-waiting) pipeline
    render_pipeline({}, pipeline_ph)

    # ── Ticker Input ──────────────────────────────────────────────────────
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        ticker = st.text_input(
            "ticker", label_visibility="collapsed",
            value=st.session_state.get("ticker", ""),
            placeholder="Enter ticker  e.g.  TCS.NS   RELIANCE.NS   AAPL   TSLA",
        ).strip().upper()
    with col_btn:
        go = st.button("RUN →")

    if not go:
        st.markdown("""
        <div style='text-align:center;padding:80px 0;'>
            <div style='font-size:3.5rem;margin-bottom:16px;'>📡</div>
            <div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;
                        color:#3d4f6e;letter-spacing:4px;'>
                ENTER A TICKER AND PRESS RUN
            </div>
        </div>""", unsafe_allow_html=True)
        return

    if not ticker:
        st.error("Please enter a stock ticker symbol.")
        return

    # ── Pipeline Execution ────────────────────────────────────────────────
    states  = {}          # step_index → "wait"|"active"|"done"|"error"
    results = {}          # collected outputs

    def tick(idx, status):
        states[idx] = status
        render_pipeline(states, pipeline_ph)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: User Input — mark done immediately
    tick(0, "done")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: Fetch Live Stock Data
    tick(1, "active")
    status_ph = st.empty()
    status_ph.info("📡 **Step 2** — Fetching live stock data from yfinance…")
    try:
        df, info = fetch_live_stock(ticker)
        results["df"]   = df
        results["info"] = info
        tick(1, "done")
        data_ts = datetime.now().strftime("%H:%M:%S")
        status_ph.success(
            f"✅ **{info['name']}** — {len(df)} trading days loaded "
            f"({df.index[0].date()} → {df.index[-1].date()})  |  fetched at {data_ts}"
        )
    except Exception as e:
        tick(1, "error")
        status_ph.error(f"❌ Stock data error: {e}")
        return

    df      = results["df"]
    info    = results["info"]
    cur     = float(df["Close"].iloc[-1])
    currency = info["currency"]
    vol, vol_label = volatility(df)

    # Quick stats row
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Current Price", fmt(cur, currency))
    c2.metric("Exchange",      info["exchange"] or "—")
    c3.metric("Sector",        info["sector"])
    c4.metric("Volatility",    f"{vol}%", vol_label)
    mktcap = info["market_cap"]
    c5.metric("Market Cap",    f"{mktcap/1e9:.1f}B {currency}" if mktcap else "—")

    st.markdown(
        f"<div class='live-badge'><span class='live-dot'></span>LIVE — data through {df.index[-1].date()}</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: Fetch Live News
    tick(2, "active")
    news_ph = st.empty()
    if not news_api_key:
        news_ph.warning("⚠️ **Step 3** — No NewsAPI key provided. Skipping news; using neutral sentiment.")
        tick(2, "done")
        results["articles"] = []
    else:
        news_ph.info(f"📰 **Step 3** — Fetching latest {max_articles} news articles…")
        try:
            articles = fetch_live_news(ticker, info["name"], news_api_key, max_articles)
            results["articles"] = articles
            tick(2, "done")
            news_ph.success(f"✅ **Step 3** — {len(articles)} live articles fetched")
        except Exception as e:
            tick(2, "error")
            news_ph.error(f"❌ News fetch error: {e}")
            results["articles"] = []

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: BERT Sentiment
    tick(3, "active")
    sent_ph = st.empty()
    articles = results["articles"]

    if not articles:
        sent_ph.info("⚪ **Step 4** — No articles to classify. Using neutral (0.5) sentiment.")
        tick(3, "done")
        results["sentiment"] = {
            "articles": [], "counts": {"positive":0,"neutral":1,"negative":0},
            "pct_pos": 0, "pct_neu": 100, "pct_neg": 0,
            "score": 0.5, "overall": "Neutral",
        }
    else:
        sent_ph.info(f"🤖 **Step 4** — Running BERT on {len(articles)} headlines…")
        t0 = time.time()
        sent = run_bert_sentiment(articles)
        results["sentiment"] = sent
        tick(3, "done")
        sent_ph.success(
            f"✅ **Step 4** — BERT done in {time.time()-t0:.1f}s  |  "
            f"Positive {sent['pct_pos']}%  Neutral {sent['pct_neu']}%  Negative {sent['pct_neg']}%  |  "
            f"Overall: **{sent['overall']}**"
        )

    # Sentiment section render
    sent = results["sentiment"]
    with st.expander("📰 Sentiment Detail", expanded=False):
        s1,s2,s3 = st.columns(3)
        s1.metric("✅ Positive", f"{sent['pct_pos']}%", f"{sent['counts']['positive']} articles")
        s2.metric("⚪ Neutral",  f"{sent['pct_neu']}%", f"{sent['counts']['neutral']} articles")
        s3.metric("🔴 Negative", f"{sent['pct_neg']}%", f"{sent['counts']['negative']} articles")
        st.markdown(f"""
        <div class='sent-bar-wrap'>
            <div style='width:{sent["pct_pos"]}%;background:#3fb950'></div>
            <div style='width:{sent["pct_neu"]}%;background:#d29922'></div>
            <div style='width:{sent["pct_neg"]}%;background:#f85149'></div>
        </div>
        """, unsafe_allow_html=True)
        for art in sent["articles"]:
            css = {"positive":"news-pos","negative":"news-neg","neutral":"news-neu"}[art["sentiment"]]
            ico = {"positive":"🟢","negative":"🔴","neutral":"🟡"}[art["sentiment"]]
            st.markdown(f"""
            <div class='news-item {css}'>
                {ico} <strong>{art['title']}</strong><br>
                <span style='color:#6e7681;font-size:0.76rem;'>
                    {art['source']} · {art['date']} · {art['sentiment'].upper()} ({art['confidence']:.0%})
                    &nbsp;<a href='{art["url"]}' target='_blank' style='color:#58a6ff;'>↗</a>
                </span>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"### 📈 Model Training & Prediction  (horizon: +{horizon} days)")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5: Feature Engineering
    tick(4, "active")
    feat_ph = st.empty()
    feat_ph.info("🔧 **Step 5** — Engineering features from live data…")
    feat = engineer_features(df)
    tick(4, "done")
    feat_ph.success(f"✅ **Step 5** — Features ready: {list(feat.columns)} ({len(feat)} rows)")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 6: Linear Regression
    tick(5, "active")
    lr_ph = st.empty()
    lr_ph.info("📐 **Step 6** — Training Linear Regression on live data…")
    t0 = time.time()
    lr_res = train_linear_regression(feat, horizon)
    results["lr"] = lr_res
    tick(5, "done")
    lr_ph.success(
        f"✅ **Step 6** — Linear Regression trained in {time.time()-t0:.1f}s  |  "
        f"Predicted price: **{fmt(lr_res['predicted'], currency)}**  |  MAE: {lr_res['mae']}"
    )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 7: Random Forest
    tick(6, "active")
    rf_ph = st.empty()
    rf_ph.info("🌲 **Step 7** — Training Random Forest (200 trees) on live data…")
    t0 = time.time()
    rf_res = train_random_forest(feat, horizon)
    results["rf"] = rf_res
    tick(6, "done")
    rf_ph.success(
        f"✅ **Step 7** — Random Forest trained in {time.time()-t0:.1f}s  |  "
        f"Predicted price: **{fmt(rf_res['predicted'], currency)}**  |  MAE: {rf_res['mae']}"
    )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 8: LSTM
    if skip_lstm:
        tick(7, "done")
        lstm_res = {"model":"LSTM (skipped)","predicted":rf_res["predicted"],"mae":rf_res["mae"]}
        st.info("⏩ **Step 8** — LSTM skipped (toggle off in sidebar).")
    else:
        tick(7, "active")
        lstm_ph = st.empty()
        lstm_ph.info(f"🧠 **Step 8** — Training LSTM ({lstm_epochs} epochs) on live data…")
        t0 = time.time()
        lstm_res = train_lstm(df, horizon, lstm_epochs)
        results["lstm"] = lstm_res
        tick(7, "done")
        lstm_ph.success(
            f"✅ **Step 8** — LSTM trained in {time.time()-t0:.1f}s  |  "
            f"Predicted price: **{fmt(lstm_res['predicted'], currency)}**  |  MAE: {lstm_res['mae']}"
        )

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 9: Ensemble
    tick(8, "active")
    ens_ph = st.empty()
    ens_ph.info("⚗️ **Step 9** — Computing MAE-weighted ensemble…")
    ens = ensemble(lr_res, rf_res, lstm_res, cur)
    results["ensemble"] = ens
    tick(8, "done")
    ens_ph.success(
        f"✅ **Step 9** — Ensemble prediction: **{fmt(ens['final'], currency)}**  |  "
        f"Change: {ens['pct_change']:+.2f}%  |  Confidence: {ens['confidence']}%"
    )

    # Model comparison table
    st.markdown("#### Model Comparison")
    st.markdown("""
    <div class='model-row header'>
        <div>MODEL</div><div>PREDICTED PRICE</div><div>MAE (test set)</div><div>WEIGHT BASIS</div>
    </div>
    """, unsafe_allow_html=True)
    for r in [lr_res, rf_res, lstm_res]:
        arrow = "🔼" if r["predicted"] >= cur else "🔽"
        st.markdown(f"""
        <div class='model-row'>
            <div style='color:#8b949e'>{r['model']}</div>
            <div style='color:#e6edf3'>{arrow} {fmt(r['predicted'], currency)}</div>
            <div style='color:#8b949e'>{r['mae']}</div>
            <div style='color:#3d4f6e'>1/MAE</div>
        </div>""", unsafe_allow_html=True)

    # Ensemble summary
    e1,e2,e3,e4 = st.columns(4)
    e1.metric("Current Price",      fmt(cur, currency))
    e2.metric(f"Ensemble (+{horizon}d)", fmt(ens["final"], currency))
    delta_sign = "+" if ens["pct_change"] >= 0 else ""
    e3.metric("Expected Change",     f"{delta_sign}{ens['pct_change']}%")
    e4.metric("Confidence",          f"{ens['confidence']}%")

    st.markdown(f"""
    <div style='margin:10px 0'>
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#3d4f6e;letter-spacing:2px;'>
            CONFIDENCE
        </div>
        <div class='conf-bar-wrap'>
            <div class='conf-bar-fill' style='width:{ens["confidence"]}%'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Price chart
    with st.expander("📉 Historical Price (Last 1Y)", expanded=True):
        st.line_chart(df[["Close"]].tail(252), use_container_width=True, color="#1f6feb")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 10: Signal
    tick(9, "active")
    sig_ph = st.empty()
    sig_ph.info("🤖 **Step 10** — Generating final signal…")
    decision = make_signal(ens, sent)
    sig       = decision["signal"]
    tick(9, "done")
    sig_ph.success(f"✅ **Step 10** — Signal: **{sig}**")

    st.markdown("### 🎯 Final AI Signal")
    st.markdown(f"""
    <div class='signal-box signal-{sig}'>
        <div class='signal-word-{sig}'>{sig}</div>
        <div class='signal-reason'>{decision['reason']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Full summary
    st.markdown("#### 📋 Full Run Summary")
    summary_data = {
        "Ticker":                  ticker,
        "Company":                 info["name"],
        "Data Range":              f"{df.index[0].date()} → {df.index[-1].date()}",
        "Current Price":           fmt(cur, currency),
        "Linear Regression Pred":  fmt(lr_res["predicted"], currency),
        "Random Forest Pred":      fmt(rf_res["predicted"], currency),
        "LSTM Pred":               fmt(lstm_res["predicted"], currency),
        f"Ensemble Pred (+{horizon}d)": fmt(ens["final"], currency),
        "Expected Growth":         f"{ens['pct_change']:+.2f}%",
        "Model Confidence":        f"{ens['confidence']}%",
        "Annualised Volatility":   f"{vol}% ({vol_label})",
        "Sentiment Score":         f"{sent['score']:.2f} / 1.00  ({sent['overall']})",
        "Articles Analysed":       str(len(articles)),
        "🤖 AI Signal":             sig,
        "Reason":                  decision["reason"],
        "Run Timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.table(
        pd.DataFrame(summary_data.items(), columns=["Metric","Value"]).set_index("Metric")
    )

    st.caption(
        "⚠️ **Disclaimer**: Educational / research use only. Not financial advice. "
        "Past performance does not guarantee future results."
    )


if __name__ == "__main__":
    main()
