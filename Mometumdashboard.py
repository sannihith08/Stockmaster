# ================================================================
# ⚡ Momentum Dashboard (VWROC + VPI + RSI + Volume + Weekly/Monthly Filters)
# Author: Sanni | Version: Oct 2025
# ================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import load_daily, enrich_daily

st.set_page_config(page_title="Momentum Dashboard", layout="wide")
st.title("⚡ Full Momentum + Weekly/Monthly Breakout Dashboard")

symbols = st.session_state.get("SYMBOLS", [])
if not symbols:
    st.warning("Upload symbols on Home page first.")
    st.stop()

st.sidebar.header("📊 Filters")
lookback = st.sidebar.slider("Momentum Lookback (days)", 5, 20, 10)
min_vol_ratio = st.sidebar.slider("Min Volume Ratio (vs 20d avg)", 0.8, 3.0, 1.2)
momo_type = st.sidebar.multiselect(
    "Show Momentum Type",
    ["🔥 Strong Up","📈 Improving","⚠️ Strong Down","📉 Weakening","➡️ Mixed"],
    default=["🔥 Strong Up","📈 Improving"]
)
vol_trend_filter = st.sidebar.multiselect(
    "Volume Trend", ["📈 Increasing","📉 Decreasing","➡️ Stable"], default=[]
)
breakout_filter = st.sidebar.multiselect(
    "Breakout Filters",
    ["Near_Week_High","Near_Week_Low","Near_Month_High","Near_Month_Low","Close>Yday_High","Close<Yday_Low"],
    default=[]
)

rows = []
progress = st.progress(0.0)

for i, s in enumerate(symbols):
    try:
        d = enrich_daily(load_daily(s))
        if len(d) < 30:
            progress.progress((i+1)/len(symbols))
            continue

        t = d.iloc[-1]
        y = d.iloc[-2]

        # -------------------------------
        # 🔹 Basic momentum & RSI
        # -------------------------------
        delta_vwroc = t["VWROC"] - y["VWROC"]
        delta_vpi   = t["VPI"] - y["VPI"]

        delta = d["Close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = float(rsi.iloc[-1])

        ema_slope = ((t["EMA7"] - y["EMA7"]) / y["EMA7"]) * 100
        vol_strength = t["Volume"] / (t["AvgVol20"] + 1e-9)

        # -------------------------------
        # 🔹 Volume trend
        # -------------------------------
        vol_5d_avg = d["Volume"].tail(5).mean()
        vol_20d_avg = d["Volume"].tail(20).mean()
        vol_change_5d = ((vol_5d_avg - vol_20d_avg) / vol_20d_avg) * 100

        if vol_change_5d > 10:
            vol_trend = "📈 Increasing"
        elif vol_change_5d < -10:
            vol_trend = "📉 Decreasing"
        else:
            vol_trend = "➡️ Stable"

        # -------------------------------
        # 🔹 Weekly / Monthly High-Low
        # -------------------------------
        week_high = d["High"].tail(5).max()
        week_low  = d["Low"].tail(5).min()
        month_high = d["High"].tail(21).max()
        month_low  = d["Low"].tail(21).min()
        yday_high = y["High"]
        yday_low  = y["Low"]

        near_week_high = abs(t["Close"] - week_high) <= 0.01 * week_high
        near_week_low  = abs(t["Close"] - week_low)  <= 0.01 * week_low
        near_month_high = abs(t["Close"] - month_high) <= 0.01 * month_high
        near_month_low  = abs(t["Close"] - month_low)  <= 0.01 * month_low

        # -------------------------------
        # 🔹 Unified Score (weighted)
        # -------------------------------
        score = (
            (t["VWROC"] * 100)
            + (t["VPI"] * 50)
            + (delta_vwroc * 40)
            + (delta_vpi * 30)
            + (ema_slope * 10)
            + ((vol_strength - 1) * 20)
            - abs(50 - rsi_val) / 5
            + (5 if vol_trend == "📈 Increasing" else (-3 if vol_trend == "📉 Decreasing" else 0))
        )

        # Label classification
        if t["Momentum_Strength"] == "🔥 Strong Up" or (score > 8 and delta_vwroc > 0):
            label = "🔥 Strong Up"
        elif score > 4:
            label = "📈 Improving"
        elif t["Momentum_Strength"] == "⚠️ Strong Down" or score < -8:
            label = "⚠️ Strong Down"
        elif score < -3:
            label = "📉 Weakening"
        else:
            label = "➡️ Mixed"

        if vol_strength < min_vol_ratio:
            progress.progress((i+1)/len(symbols))
            continue

        rows.append({
            "Symbol": s,
            "Close": round(t["Close"],2),
            "VWROC": round(t["VWROC"],4),
            "ΔVWROC": round(delta_vwroc,4),
            "VPI": round(t["VPI"],4),
            "ΔVPI": round(delta_vpi,4),
            "RSI": round(rsi_val,2),
            "EMA_Slope(%)": round(ema_slope,2),
            "Vol/AvgVol": round(vol_strength,2),
            "Vol_Change_5d(%)": round(vol_change_5d,2),
            "Vol_Trend": vol_trend,
            "Yday_High": round(yday_high,2),
            "Yday_Low": round(yday_low,2),
            "Week_High": round(week_high,2),
            "Week_Low": round(week_low,2),
            "Month_High": round(month_high,2),
            "Month_Low": round(month_low,2),
            "Near_Week_High": near_week_high,
            "Near_Week_Low": near_week_low,
            "Near_Month_High": near_month_high,
            "Near_Month_Low": near_month_low,
            "Close>Yday_High": t["Close"] > yday_high,
            "Close<Yday_Low": t["Close"] < yday_low,
            "Momentum_Strength": t["Momentum_Strength"],
            "Momentum_Trend": t["Momentum_Trend"],
            "Label": label,
            "Score": round(score,2)
        })
    except Exception:
        pass

    progress.progress((i+1)/len(symbols))

df = pd.DataFrame(rows)
if df.empty:
    st.warning("No data generated.")
    st.stop()

# ---------------------------------------------------------
# 🔎 Apply Filters
# ---------------------------------------------------------
mask = pd.Series(True, index=df.index)
if momo_type:
    mask &= df["Label"].isin(momo_type)
if vol_trend_filter:
    mask &= df["Vol_Trend"].isin(vol_trend_filter)
if breakout_filter:
    for bf in breakout_filter:
        if bf in df.columns:
            mask &= df[bf] == True

view = df[mask].sort_values("Score", ascending=False)

# ---------------------------------------------------------
# 🧮 Display
# ---------------------------------------------------------
st.subheader("📊 Filtered Momentum Stocks")
st.dataframe(
    view.head(50)
    .style.background_gradient(cmap="Greens", subset=["Score"])
    .highlight_max(color="lightgreen", subset=["VWROC","VPI"])
    .highlight_between(subset=["RSI"], left=40, right=60, color="#FFFACD"),
    use_container_width=True
)

# ---------------------------------------------------------
# 📈 Visualization
# ---------------------------------------------------------
fig = px.scatter(
    df,
    x="VWROC", y="VPI",
    color="Vol_Trend",
    size=df["Score"].abs(),  # ✅ only positive sizes
    hover_data=["Symbol","Label","RSI","Vol/AvgVol","Near_Week_High","Near_Month_High"],
    title="Momentum Map (VWROC vs VPI, colored by Volume Trend)",
    template="plotly_dark"
)

# ---------------------------------------------------------
# 📥 Download
# ---------------------------------------------------------
st.download_button(
    "📥 Download Momentum Data",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="momentum_dashboard_full.csv",
    mime="text/csv"
)
