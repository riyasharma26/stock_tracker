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
if "ticker_input" not in st.session_state:
    st.session_state.ticker_input = ""
if "charts" not in st.session_state:
    st.session_state.charts = {}

tabs = st.tabs(["📊 Portfolio", "📈 Weekly Picks", "ℹ️ How It Works"])

# --- Tab 1: Portfolio ---
with tabs[0]:
    # Manual input form
    with st.form("manual_input"):
        st.subheader("Add a Stock to Your Portfolio")
        st.session_state.ticker_input = st.text_input("Ticker (e.g. AAPL)", value=st.session_state.ticker_input).upper()
        shares_input = st.number_input("Shares Owned", min_value=0.0001, format="%.4f")
        submitted = st.form_submit_button("Add to Portfolio")

        if submitted:
            ticker_input = st.session_state.ticker_input
            if ticker_input and shares_input > 0:
                new_row = pd.DataFrame({"Ticker": [ticker_input], "Shares": [shares_input]})
                if ticker_input in st.session_state.portfolio["Ticker"].values:
                    idx = st.session_state.portfolio[st.session_state.portfolio["Ticker"] == ticker_input].index[0]
                    st.session_state.portfolio.at[idx, "Shares"] += shares_input
                else:
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                st.success(f"Added {shares_input} shares of {ticker_input} to portfolio")
                st.session_state.ticker_input = ""  # Clear input

    # CSV Upload
    uploaded_file = st.file_uploader("Or upload your portfolio CSV (Ticker, Shares)", type=["csv"])
    if uploaded_file:
        uploaded_portfolio = pd.read_csv(uploaded_file)
        combined = pd.concat([st.session_state.portfolio, uploaded_portfolio], ignore_index=True)
        combined = combined.groupby("Ticker", as_index=False).agg({"Shares": "sum"})
        st.session_state.portfolio = combined

    # Current Portfolio Table and Insights
    if not st.session_state.portfolio.empty:
        st.subheader("📈 Portfolio with Buy/Sell Signals")

        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)
        projections = []
        st.session_state.charts = {}

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

                # Store chart
                fig, ax = plt.subplots()
                ax.plot(hist.index, hist["Close"], label="Close", color="blue")
                ax.plot(hist.index, hist["Close"].rolling(window=50).mean(), label="50-Day MA", color="orange")
                ax.plot(hist.index, hist["Close"].rolling(window=200).mean(), label="200-Day MA", color="purple")
                ax.axhline(est_buy, color='green', linestyle='--', label='Est. Buy')
                ax.axhline(est_sell, color='red', linestyle='--', label='Est. Sell')
                ax.set_title(f"{ticker} — Trend & Thresholds")
                ax.legend()
                st.session_state.charts[ticker] = fig

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

            # Download
            buffer = io.StringIO()
            proj_df.to_csv(buffer, index=False)
            st.download_button(
                label="Download Insights CSV",
                data=buffer.getvalue(),
                file_name="portfolio_insights.csv",
                mime="text/csv"
            )

            # Dropdown to view charts
            st.subheader("📊 View Stock Chart")
            selected_chart = st.selectbox("Choose a stock to view chart", list(st.session_state.charts.keys()))
            if selected_chart:
                st.pyplot(st.session_state.charts[selected_chart])
    else:
        st.info("Add stocks manually above or upload a CSV to get started.")

# --- Tab 2: Weekly Picks ---
with tabs[1]:
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

# --- Tab 3: How It Works ---
with tabs[2]:
    st.subheader("📘 How It Works")
    st.markdown("""
    ## What is the Intelligent Stock Portfolio Tracker?

    This tool helps you:
    - **Track your holdings**
    - **See predictions and trends**
    - **Discover new investment opportunities**

    ---

    ## 🧠 How It Works

    1. **Data Fetching** from Yahoo Finance
    2. **Moving Averages** (50 & 200-day)
    3. **Trend Forecasting** using Linear Regression
    4. **Future Growth Simulation** over 1, 3, 5 years at 8% CAGR

    ---

    ## 📋 How to Use It

    1. **Add stocks manually** or upload CSV
    2. **See projections**, price charts, and trends
    3. **Check the “Weekly Picks”** for new opportunities
    4. **Download insights** for deeper analysis

    ---

    **Tip:** Revisit weekly for updated forecasts and buy signals.
    """)
