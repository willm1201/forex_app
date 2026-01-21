# Forex AI Decision Engine – Profit-Optimized (Paper Trading Only)
# EDUCATIONAL USE ONLY – NOT FINANCIAL ADVICE
# Mode: Trade Recommendation + Paper Trading + Strict Risk Controls

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------- APP CONFIG ----------------
st.set_page_config(page_title="Forex AI Profit Engine", layout="wide")
st.title("💱 Forex AI Profit Optimization Engine")
st.warning("Decision-support only. AI does NOT execute trades or guarantee profit. TEST")

# ---------------- API KEYS ----------------
# Preferred: Streamlit Secrets (works on localhost & Streamlit Cloud)
# Create .streamlit/secrets.toml with:
# FX_API_KEY = "O27B69SGSTW7BT40"
FX_API_KEY = st.secrets.get("O27B69SGSTW7BT40", "")

# Optional fallback: environment variable
import os
if not FX_API_KEY:
    FX_API_KEY = os.getenv("FX_API_KEY", "")

if not FX_API_KEY:
    st.info("FX_API_KEY not set. Add it to .streamlit/secrets.toml or as an environment variable.")

# ---------------- RISK LIMITS (HARD CONSTRAINTS) ----------------
MAX_RISK_PER_TRADE = 0.01     # 1%
MAX_DAILY_LOSS = 0.03         # 3%
MAX_CONSECUTIVE_LOSSES = 3

# ---------------- SESSION STATE ----------------
if 'equity' not in st.session_state:
    st.session_state.equity = 10000.0
    st.session_state.daily_loss = 0.0
    st.session_state.loss_streak = 0
    st.session_state.trade_log = []

# ---------------- USER INPUT ----------------
PAIR = st.selectbox("Currency Pair", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"])
TF = st.selectbox("Timeframe", ["5m", "15m", "1h"])

# ---------------- DATA INGESTION ----------------
# NOTE: Replace example endpoint with your real FX data provider
# Examples: Alpha Vantage, Twelve Data, Polygon, OANDA (read-only)

def fetch_fx(pair, timeframe):
    """
    Polygon.io FX aggregates
    Docs: https://polygon.io/docs/forex
    """
    symbol = pair.replace("/", "")

    # Map Streamlit timeframe to Polygon multiplier/timespan
    tf_map = {
        "5m": (5, "minute"),
        "15m": (15, "minute"),
        "1h": (1, "hour"),
    }
    multiplier, timespan = tf_map[timeframe]

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/C:{symbol}/range/"
        f"{multiplier}/{timespan}/"
        f"{(datetime.utcnow().date().replace(day=1))}/"
        f"{datetime.utcnow().date()}"
        f"?adjusted=true&sort=asc&limit=5000&apiKey={FX_API_KEY}"
    )

    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None

    data = r.json()
    if 'results' not in data:
        return None

    df = pd.DataFrame(data['results'])
    df.rename(columns={
        't': 'datetime',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close'
    }, inplace=True)

    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    return df[['datetime', 'open', 'high', 'low', 'close']]

# ---------------- LOAD DATA ----------------
if st.button("Run AI Analysis"):
    if not FX_API_KEY:
        st.error("Missing FX_API_KEY. Set it before running analysis.")
        st.stop()

    df = fetch_fx(PAIR, TF)
    if df is None or df.empty:
        st.error("FX provider not connected or returned no data.")
        st.stop()

    df = df.sort_values("datetime")

    # ---------------- FEATURES ----------------
    df['return'] = df['close'].pct_change()
    df['range'] = (df['high'] - df['low']) / df['close']
    df['atr'] = df['range'].rolling(14).mean()
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['ma_fast'] = df['close'].rolling(10).mean()
    df['ma_slow'] = df['close'].rolling(30).mean()
    df.dropna(inplace=True)

    # ---------------- TARGET ----------------
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    X = df[['return','range','atr','momentum','ma_fast','ma_slow']]
    y = df['target']

    split = int(len(df) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42))
    ])

    pipe.fit(X_train, y_train)

    # ---------------- AI DECISION ----------------
    latest = X.iloc[-1:]
    proba = pipe.predict_proba(latest)[0]
    direction = 1 if proba[1] > 0.55 else 0

    win_rate = proba[direction]
    rr = 1.5
    ev = (win_rate * rr) - (1 - win_rate)

    # ---------------- RISK GATE ----------------
    if st.session_state.daily_loss > MAX_DAILY_LOSS * st.session_state.equity:
        st.error("AI DISABLED: Daily loss limit reached")
        st.stop()

    if st.session_state.loss_streak >= MAX_CONSECUTIVE_LOSSES:
        st.error("AI DISABLED: Loss streak limit reached")
        st.stop()

    if ev <= 0:
        st.info("NO TRADE: Negative expected value")
        st.stop()

    # ---------------- TRADE RECOMMENDATION ----------------
    risk_amount = st.session_state.equity * MAX_RISK_PER_TRADE
    stop_pips = df['atr'].iloc[-1] * 10000 * 1.2
    position = risk_amount / max(stop_pips, 1)

    st.subheader("AI Trade Recommendation")
    st.success(f"{'BUY' if direction == 1 else 'SELL'} {PAIR}")
    st.write(f"Expected Value: {ev:.2f}")
    st.write(f"Win Probability: {win_rate:.2%}")
    st.write(f"Position Size: {position:.2f} lots")
    st.write(f"Stop Loss: {stop_pips:.1f} pips")
    st.write(f"Risk: {MAX_RISK_PER_TRADE*100:.1f}% of equity")

    # ---------------- PAPER TRADE SIMULATION ----------------
    realized_return = df['return'].iloc[-1]
    pnl = risk_amount * (rr if realized_return > 0 else -1)

    st.session_state.equity += pnl
    if pnl < 0:
        st.session_state.loss_streak += 1
        st.session_state.daily_loss += abs(pnl)
    else:
        st.session_state.loss_streak = 0

    st.session_state.trade_log.append({
        "Time": datetime.utcnow(),
        "Pair": PAIR,
        "Direction": "BUY" if direction else "SELL",
        "PnL": pnl,
        "Equity": st.session_state.equity
    })

    st.metric("Simulated Equity", f"${st.session_state.equity:,.2f}")

# ---------------- LOG ----------------
st.subheader("Paper Trade Log")
if st.session_state.trade_log:
    st.dataframe(pd.DataFrame(st.session_state.trade_log))

st.caption("AI recommends trades | Paper trading | Capital-protected")
