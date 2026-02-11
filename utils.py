# utils.py
import yfinance as yf
import pandas as pd
import numpy as np
from functools import lru_cache

# ---- Core: safe yfinance with de-multiindex ----
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    return df

@lru_cache(maxsize=2048)
def load_daily(symbol: str, period: str = "400d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    df = _flatten_cols(df).dropna(subset=["Open","High","Low","Close"])
    return df
def load_daily_52w(symbol: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        period="3y",          # ✅ enough for 52W
        interval="1d",
        auto_adjust=False,    # 🔥 MOST IMPORTANT
        progress=False
    )

    if df.empty:
        return df

    # Fix column format
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure clean numeric data
    df = df.dropna(subset=["High", "Low", "Close"])

    return df

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()

def calc_cpr(y_high, y_low, y_close):
    pivot = (y_high + y_low + y_close) / 3.0
    bc = (y_high + y_low) / 2.0
    tc = 2 * pivot - bc
    r1 = (2 * pivot) - y_low
    s1 = (2 * pivot) - y_high
    r2 = pivot + (y_high - y_low)
    s2 = pivot - (y_high - y_low)
    return pivot, bc, tc, r1, s1, r2, s2

def cpr_trend(y_pivot, y_bc, y_tc, dby_pivot, dby_bc, dby_tc):
    if (y_pivot > dby_pivot) and (y_bc > dby_bc) and (y_tc > dby_tc):
        return "Ascending"
    if (y_pivot < dby_pivot) and (y_bc < dby_bc) and (y_tc < dby_tc):
        return "Descending"
    return "Sideways"

def latest_cpr_trend(price, bc, tc):
    if price > tc: return "Ascending"
    if price < bc: return "Descending"
    return "Neutral"

def detect_swings(df: pd.DataFrame, left=2, right=2, top_n=5):
    highs, lows = [], []
    H, L = df["High"].values, df["Low"].values
    n = len(df)
    for i in range(n):
        if i < left or i + right >= n:
            highs.append(np.nan); lows.append(np.nan); continue
        window_H = H[i-left:i+right+1]
        window_L = L[i-left:i+right+1]
        highs.append(H[i] if H[i] == window_H.max() else np.nan)
        lows.append(L[i] if L[i] == window_L.min() else np.nan)
    df = df.copy()
    df["Swing_High"] = highs
    df["Swing_Low"] = lows
    df["Last_Swing_High"] = df["Swing_High"].ffill()
    df["Last_Swing_Low"] = df["Swing_Low"].ffill()
    sh = df["Swing_High"].dropna().tail(top_n).round(2).tolist()[::-1]
    sl = df["Swing_Low"].dropna().tail(top_n).round(2).tolist()[::-1]
    return df, (", ".join(map(str, sh)) if sh else "—"), (", ".join(map(str, sl)) if sl else "—")

def enrich_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA7"] = ema(df["Close"], 7)
    df["EMA20"] = ema(df["Close"], 20)

    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift()).abs()
    df["L-PC"] = (df["Low"] - df["Close"].shift()).abs()
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(14, min_periods=1).mean()
    #df["ATR_Ratio"] = df["ATR"] / df["Close"].replace(0, np.nan)
    df["ATR_Ratio"] = df["ATR"] / df["Close"].astype(float)

    df["AvgVol20"] = df["Volume"].rolling(20, min_periods=1).mean()
    df["20d_High"] = df["High"].rolling(20, min_periods=1).max()
    df["20d_Low"] = df["Low"].rolling(20, min_periods=1).min()

    # # 52W with min_periods so it doesn't go NaN early
    # df["52W_High"] = df["High"].rolling(252, min_periods=50).max()
    # df["52W_Low"]  = df["Low"].rolling(252, min_periods=50).min()

 # ✅ FIXED 52-WEEK LOGIC (true 52 weeks = 252 trading days)
    WINDOW_52W = 252
    df["52W_High"] = df["High"].rolling(window=WINDOW_52W, min_periods=WINDOW_52W).max()
    df["52W_Low"]  = df["Low"].rolling(window=WINDOW_52W, min_periods=WINDOW_52W).min()
    # momentum
    n = 5
    prev = df["Close"].shift(n)
    avg_vol_n = df["Volume"].rolling(n, min_periods=1).mean()
    df["VWROC"] = ((df["Close"] - prev) / prev.replace(0, np.nan)) * (df["Volume"] / avg_vol_n.replace(0, np.nan))

    df["UpVol"] = np.where(df["Close"] > df["Open"], df["Volume"], 0)
    df["DownVol"] = np.where(df["Close"] < df["Open"], df["Volume"], 0)
    total_vol = df["Volume"].rolling(n, min_periods=1).sum().replace(0, np.nan)
    up_sum = df["UpVol"].rolling(n, min_periods=1).sum()
    down_sum = df["DownVol"].rolling(n, min_periods=1).sum()
    df["VPI"] = (up_sum - down_sum) / total_vol

    def mclass(row):
        if row["VWROC"] > 0.05 and row["VPI"] > 0.2: return "🔥 Strong Up"
        if row["VWROC"] < -0.05 and row["VPI"] < -0.2: return "⚠️ Strong Down"
        if abs(row["VWROC"]) > 0.02: return "↔️ Mild"
        return "Neutral"
    df["Momentum_Strength"] = df.apply(mclass, axis=1)
    df["Momentum_Trend"] = np.where(
        (df["VWROC"].diff() > 0) & (df["VPI"].diff() > 0), "📈 Improving",
        np.where((df["VWROC"].diff() < 0) & (df["VPI"].diff() < 0), "📉 Weakening", "➡️ Mixed")
    )
    return df

def breakout_tags(row, y_high, y_low, y_bc, y_tc):
    tags = []
    if row["Close"] > y_tc: tags.append("CPR_Top")
    if row["Close"] < y_bc: tags.append("CPR_Bottom")
    if row["Close"] > row.get("Last_Swing_High", np.inf) * 0.999: tags.append("Swing")
    if pd.notna(row.get("52W_High")) and row["Close"] >= 0.999 * row["52W_High"]: tags.append("52W")
    return " + ".join(tags) if tags else "—"
