import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from sklearn.linear_model import LinearRegression
import numpy as np
import io

st.set_page_config(layout="wide")
st.title("Intelligent Stock Portfolio Tracker")

# Initialize session state
if "portfolio" not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Ticker", "Shares"])

tabs = st.tabs(["📊 Portfolio", "📉 Portfolio Insights", "📈 Weekly Picks", "ℹ️ How It Works"])

# --- Tab 1: Portfolio ---
with tabs[0]:
    # Manual input form
    with st.form("manual_input"):
        st.subheader("Add a Stock to Your Portfolio")
        ticker_input = st.text_input("Ticker (e.g. AAPL)").upper()
        shares_input = st.number_input("Shares Owned", min_value=0.0001, format="%.4f")
        submitted = st.form_submit_button("Add to Portfolio")

        if submitted:
            if ticker_input and shares_input > 0:
                new_row = pd.DataFrame({"Ticker": [ticker_input], "Shares": [shares_input]})
                if ticker_input in st.session_state.portfolio["Ticker"].values:
                    idx = st.session_state.portfolio[st.session_state.portfolio["Ticker"] == ticker_input].index[0]
                    st.session_state.portfolio.at[idx, "Shares"] += shares_input
                else:
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                st.success(f"Added {shares_input} shares of {ticker_input} to portfolio")

    # CSV Upload
    uploaded_file = st.file_uploader("Or upload your portfolio CSV (Ticker, Shares)", type=["csv"])
    if uploaded_file:
        uploaded_portfolio = pd.read_csv(uploaded_file)
        combined = pd.concat([st.session_state.portfolio, uploaded_portfolio], ignore_index=True)
        combined = combined.groupby("Ticker", as_index=False).agg({"Shares": "sum"})
        st.session_state.portfolio = combined

    # Current Portfolio Display
    if not st.session_state.portfolio.empty:
        st.subheader("Current Portfolio")
        for i, row in st.session_state.portfolio.iterrows():
            ticker = row["Ticker"]
            shares = row["Shares"]
            col1, col2, col3 = st.columns([3, 2, 1])
            col1.markdown(f"**{ticker}**")
            col2.markdown(f"{shares} shares")
            if col3.button("🗑️", key=f"remove_{ticker}"):
                st.session_state.portfolio = st.session_state.portfolio.drop(i).reset_index(drop=True)
                st.experimental_rerun()
    else:
        st.info("Add stocks manually above or upload a CSV to get started.")

# --- Tab 2: Portfolio Insights ---
with tabs[1]:
    st.subheader("Portfolio Insights with Predictive Thresholds")

    if not st.session_state.portfolio.empty:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)
        projections = []
        charts = []

        for index, row in st.session_state.portfolio.iterrows():
            ticker = row["Ticker"]
            shares = row["Shares"]

            try:
                stock = yf.Ticker(ticker)
                time.sleep(1.5)
                hist = stock.history(start=start_date, end=end_date)

                if hist.empty:
                    continue

                current_price = hist["Close"][-1]
                avg_50 = hist["Close"].rolling(window=50).mean().iloc[-1]
                avg_200 = hist["Close"].rolling(window=200).mean().iloc[-1]
                total_value = current_price * shares

                cagr = 0.08
                future_values = {f"{n}y": round(total_value * (1 + cagr) ** n, 2) for n in [1, 3, 5]}

                # Linear regression prediction
                hist_recent = hist[-90:].copy().reset_index()
                hist_recent["Days"] = (hist_recent["Date"] - hist_recent["Date"].min()).dt.days
                X = hist_recent[["Days"]]
                y = hist_recent["Close"]
                model = LinearRegression().fit(X, y)
                current_day = hist_recent["Days"].max()
                predicted_today = model.predict([[current_day]])[0]
                predicted_in_30 = model.predict([[current_day + 30]])[0]
                est_buy = round(predicted_today * 0.95, 2)
                est_sell = round(predicted_in_30, 2)

                signal = "BUY" if avg_50 > avg_200 else "HOLD"
                projections.append({
                    "Ticker": ticker,
                    "Current Price": round(current_price, 2),
                    "Total Value": round(total_value, 2),
                    "50-Day MA": round(avg_50, 2),
                    "200-Day MA": round(avg_200, 2),
                    "Signal": signal,
                    "Est. Buy Price": est_buy,
                    "Est. Sell Price": est_sell,
                    **future_values
                })

                # Chart for each stock
                fig, ax = plt.subplots()
                ax.plot(hist.index, hist["Close"], label="Close Price", color="blue")
                ax.plot(hist.index, hist["Close"].rolling(window=50).mean(), label="50-Day MA", color="orange")
                ax.plot(hist.index, hist["Close"].rolling(window=200).mean(), label="200-Day MA", color="purple")
                ax.axhline(est_buy, color='green', linestyle='--', label='Est. Buy')
                ax.axhline(est_sell, color='red', linestyle='--', label='Est. Sell')
                ax.set_title(f"{ticker} — Price Trend & Buy/Sell Zones")
                ax.legend()
                charts.append((ticker, fig))

            except Exception as e:
                st.warning(f"Could not process {ticker}: {e}")
                continue

        if projections:
            proj_df = pd.DataFrame(projections)

            def color_signal(val):
                color = 'green' if val == 'BUY' else 'red'
                return f'color: {color}; font-weight: bold;'

            styled_df = proj_df.style.applymap(color_signal, subset=["Signal"])
            st.dataframe(styled_df, use_container_width=True)

            buffer = io.StringIO()
            proj_df.to_csv(buffer, index=False)
            st.download_button(
                label="Download Insights CSV",
                data=buffer.getvalue(),
                file_name="portfolio_insights.csv",
                mime="text/csv"
            )

            # Show charts in new sub-tab
            chart_tab = st.tabs(["📊 Stock Charts"])[0]
            with chart_tab:
                for ticker, fig in charts:
                    st.write(f"**{ticker}**")
                    st.pyplot(fig)

    else:
        st.info("Add stocks in the Portfolio tab first to view insights.")

# --- Tab 3: Weekly Picks ---
with tabs[2]:
    st.subheader("Weekly Growth Picks Outside Your Portfolio")
    suggestions = ["NVDA", "SMCI", "MELI", "CRWD", "TSLA"]
    current_tickers = st.session_state.portfolio["Ticker"].tolist()
    for ticker in suggestions:
        if ticker not in current_tickers:
            st.markdown(f"**{ticker}** — Potential momentum stock")
            if st.button(f"Add {ticker}", key=f"add_{ticker}"):
                new_row = pd.DataFrame({"Ticker": [ticker], "Shares": [0]})
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                st.success(f"{ticker} added to portfolio!")

# --- Tab 4: How It Works ---
with tabs[3]:
    st.subheader("📘 How It Works")
    st.markdown("""
**Welcome to the Intelligent Stock Portfolio Tracker!**

This tool helps you:
- ✅ Track and visualize your personal stock portfolio
- 📈 View predictive buy/sell price thresholds based on trendlines
- 💡 Get weekly ideas for new growth stocks
- 📊 Download insights in CSV format

---

### 🔧 How to Use
1. Go to **Portfolio tab** to manually add tickers or upload a CSV.
2. Switch to **Portfolio Insights** to:
   - See buy/sell recommendations
   - View future projections
   - Download your insights
3. Visit **Weekly Picks** for stocks outside your portfolio that may grow.
4. Use **🗑️** icons to remove tickers instantly.

---

### 📡 Methodology
- Uses **Yahoo Finance** data
- Calculates 50-day and 200-day moving averages
- Applies **linear regression** for near-future forecasting
- Gives BUY/HOLD signal based on momentum

---

Try adding a few stocks like `AAPL`, `GOOG`, or `TSLA` and explore your smart dashboard!
    """)
