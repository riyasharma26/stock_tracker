import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

st.title("Intelligent Stock Portfolio Tracker")

# Initialize session state to hold portfolio data
if "portfolio" not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Ticker", "Shares"])

# Manual input form
with st.form("manual_input"):
    st.write("### Add a Stock to Your Portfolio")
    ticker_input = st.text_input("Ticker (e.g. AAPL)").upper()
    shares_input = st.number_input("Shares Owned", min_value=1, step=1)
    submitted = st.form_submit_button("Add to Portfolio")

    if submitted:
        if ticker_input and shares_input > 0:
            new_row = pd.DataFrame({"Ticker": [ticker_input], "Shares": [shares_input]})
            # Avoid duplicate tickers: update shares if ticker exists
            if ticker_input in st.session_state.portfolio["Ticker"].values:
                idx = st.session_state.portfolio[st.session_state.portfolio["Ticker"] == ticker_input].index[0]
                st.session_state.portfolio.at[idx, "Shares"] += shares_input
            else:
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
            st.success(f"Added {shares_input} shares of {ticker_input} to portfolio")

# CSV upload
uploaded_file = st.file_uploader("Or upload your portfolio CSV (Ticker, Shares)", type=["csv"])
if uploaded_file:
    uploaded_portfolio = pd.read_csv(uploaded_file)
    # Merge uploaded portfolio with manual input, summing shares for duplicates
    combined = pd.concat([st.session_state.portfolio, uploaded_portfolio], ignore_index=True)
    combined = combined.groupby("Ticker", as_index=False).agg({"Shares": "sum"})
    st.session_state.portfolio = combined

# Show current portfolio
if not st.session_state.portfolio.empty:
    st.write("### Your Current Portfolio")
    st.dataframe(st.session_state.portfolio)

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    projections = []

    for index, row in st.session_state.portfolio.iterrows():
        ticker = row["Ticker"]
        shares = row["Shares"]

        try:
            stock = yf.Ticker(ticker)
            time.sleep(1.5)  # Add delay to avoid rate limiting
            hist = stock.history(start=start_date, end=end_date)

            if hist.empty:
                st.warning(f"No data found for {ticker}")
                continue

            current_price = hist["Close"][-1]
            avg_50 = hist["Close"].rolling(window=50).mean().iloc[-1]
            avg_200 = hist["Close"].rolling(window=200).mean().iloc[-1]
            total_value = current_price * shares

            # Projected growth (assume 8% annually)
            cagr = 0.08
            future_values = {f"{n}y": round(total_value * (1 + cagr) ** n, 2) for n in [1, 3, 5]}

            projections.append({
                "Ticker": ticker,
                "Current Price": round(current_price, 2),
                "Total Value": round(total_value, 2),
                "50-Day MA": round(avg_50, 2),
                "200-Day MA": round(avg_200, 2),
                "Signal": "BUY" if avg_50 > avg_200 else "HOLD",
                **future_values
            })

            # Price chart
            st.write(f"#### {ticker} - Historical Price")
            fig, ax = plt.subplots()
            ax.plot(hist.index, hist["Close"], label="Close Price")
            ax.plot(hist.index, hist["Close"].rolling(window=50).mean(), label="50-Day MA")
            ax.plot(hist.index, hist["Close"].rolling(window=200).mean(), label="200-Day MA")
            ax.set_title(f"{ticker} Price Trend")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            continue

    if projections:
        proj_df = pd.DataFrame(projections)
        st.write("### 💡 Portfolio Insights")
        st.dataframe(proj_df)
else:
    st.info("Add stocks manually above or upload a CSV to get started!")
