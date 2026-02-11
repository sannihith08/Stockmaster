# app.py
import streamlit as st
import pandas as pd

import streamlit as st
import hashlib

# ---------------------------
# 🔒 Simple Login System
# ---------------------------
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# Store usernames & hashed passwords
usernames = ["sanni", "sunny", "Admin"]
passwords = [make_hashes("Sunny@123"), make_hashes("Data@123"), make_hashes("Admin@123")]

st.sidebar.title("🔐 Login")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if username in usernames:
    user_index = usernames.index(username)
    if check_hashes(password, passwords[user_index]):
        st.sidebar.success(f"Welcome {username} 👋")
    else:
        st.error("❌ Incorrect password")
        st.stop()
else:
    st.warning("Please login to continue")
    st.stop()


st.set_page_config(page_title="Stock Master", layout="wide")
st.title("📊 Stock Master")
st.caption("Multi-page scanner with CPR, Swing, 52W, Momentum, and Gainers/Losers")

st.sidebar.header("📂 Upload Symbols")
uploaded = st.sidebar.file_uploader("Upload CSV with 'Symbol' column", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        symbols = df["Symbol"].dropna().astype(str).unique().tolist()
        st.session_state["SYMBOLS"] = symbols
        st.success(f"Loaded {len(symbols)} symbols.")
        st.dataframe(pd.DataFrame({"Symbol": symbols}).head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Error reading file: {e}")

st.markdown("""
### ✅ What’s inside
- **Breakout Scanner**: Swing/CPR/52W breakouts, momentum, volume filters, scoring
- **Top Gainers & Losers**: Daily % movers from your list
- **Near 52W High/Low**: Within ±3% of 52W extremes
- **Momentum Dashboard**: VWROC + VPI trends, improving vs. weakening

> Tip: Upload symbols first, then open pages from the sidebar.
""")
