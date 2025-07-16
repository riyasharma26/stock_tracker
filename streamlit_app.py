import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.title("Intelligent Stock Portfolio Tracker")

# Upload CSV
uploaded_file = st.file_uploader("Upload your portfolio CSV (Ticker, Shares)", type=["csv"])
if uploaded_file:
    portfolio = pd.read_csv(uploaded_file)
    st.write("### Your Portfolio", portfolio)

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    projections = []

    for index, row in portfolio.iterrows():
        ticker = row["Ticker"]
        shares = row["Shares"]
        stock = yf.Ticker(ticker)
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

    # Show summary table
    if projections:
        proj_df = pd.DataFrame(projections)
        st.write("### 💡 Portfolio Insights")
        st.dataframe(proj_df)
