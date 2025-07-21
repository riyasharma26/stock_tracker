import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Stock Portfolio Tracker", layout="wide")
st.title("Intelligent Stock Portfolio Tracker")

# Initialize portfolio
if "portfolio" not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Ticker", "Shares"])

tabs = st.tabs(["📊 Portfolio", "📈 Weekly Picks", "ℹ️ How It Works"])

# ============================
# 📊 Portfolio Tab
# ============================
with tabs[0]:
    col1, col2 = st.columns([1, 2])

    with col1:
        with st.form("manual_input"):
            st.subheader("Add a Stock")
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

        uploaded_file = st.file_uploader("Or upload CSV (Ticker, Shares)", type=["csv"])
        if uploaded_file:
            uploaded_portfolio = pd.read_csv(uploaded_file)
            combined = pd.concat([st.session_state.portfolio, uploaded_portfolio], ignore_index=True)
            st.session_state.portfolio = combined.groupby("Ticker", as_index=False).agg({"Shares": "sum"})

    with col2:
        st.subheader("Current Portfolio")
        portfolio_df = st.session_state.portfolio.copy()
        for idx, row in portfolio_df.iterrows():
            colA, colB, colC = st.columns([2, 2, 1])
            with colA:
                st.write(f"**{row['Ticker']}** - {row['Shares']} shares")
            with colC:
                if st.button("🗑️", key=f"remove_{idx}"):
                    st.session_state.portfolio.drop(index=idx, inplace=True)
                    st.session_state.portfolio.reset_index(drop=True, inplace=True)
                    st.experimental_rerun()

    if not st.session_state.portfolio.empty:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)
        projections = []

        for index, row in st.session_state.portfolio.iterrows():
            ticker = row["Ticker"]
            shares = row["Shares"]
            try:
                stock = yf.Ticker(ticker)
                time.sleep(1.5)
                hist = stock.history(start=start_date, end=end_date)

                if hist.empty:
                    st.warning(f"No data for {ticker}")
                    continue

                current_price = hist["Close"][-1]
                avg_50 = hist["Close"].rolling(window=50).mean().iloc[-1]
                avg_200 = hist["Close"].rolling(window=200).mean().iloc[-1]
                total_value = current_price * shares
                cagr = 0.08
                future_values = {f"{n}y": round(total_value * (1 + cagr) ** n, 2) for n in [1, 3, 5]}
                signal = "BUY" if avg_50 > avg_200 else "HOLD"
                color = "green" if signal == "BUY" else "red"

                # Forecast
                recent = hist[-90:].reset_index()
                recent["Day"] = np.arange(len(recent))
                model = LinearRegression()
                model.fit(recent[["Day"]], recent["Close"])
                next_30 = model.predict(np.array([[len(recent) + 30]]))[0]
                buy_threshold = current_price * 0.95

                projections.append({
                    "Ticker": ticker,
                    "Current Price": round(current_price, 2),
                    "Total Value": round(total_value, 2),
                    "50-Day MA": round(avg_50, 2),
                    "200-Day MA": round(avg_200, 2),
                    "Signal": signal,
                    "Buy Threshold": round(buy_threshold, 2),
                    "30-Day Forecast": round(next_30, 2),
                    **future_values
                })

                # Plot
                st.subheader(f"{ticker} Price Trend")
                fig, ax = plt.subplots()
                ax.plot(hist.index, hist["Close"], label="Close Price", color="black")
                ax.plot(hist.index, hist["Close"].rolling(window=50).mean(), label="50-Day MA", color="blue")
                ax.plot(hist.index, hist["Close"].rolling(window=200).mean(), label="200-Day MA", color="orange")
                ax.axhline(buy_threshold, linestyle="--", color="green", label="Buy Threshold")
                ax.axhline(next_30, linestyle="--", color="red", label="30-Day Forecast")
                ax.set_title(f"{ticker} - Historical Chart")
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error fetching {ticker}: {e}")
                continue

        if projections:
            df_proj = pd.DataFrame(projections)
            df_proj["Signal"] = df_proj["Signal"].apply(lambda x: f":green[{x}]" if x == "BUY" else f":red[{x}]")
            st.subheader("Portfolio Insights")
            st.dataframe(df_proj, use_container_width=True)
            csv_download = df_proj.to_csv(index=False).encode("utf-8")
            st.download_button("Download Insights CSV", csv_download, file_name="portfolio_insights.csv", mime="text/csv")
    else:
        st.info("Add stocks to start tracking.")

# ============================
# 📈 Weekly Picks Tab
# ============================
with tabs[1]:
    st.title("Weekly Trending Picks")

    suggestions = ["NVDA", "MSFT", "GOOGL", "TSLA", "META"]
    st.write("These trending stocks had strong performance this week and aren't in your portfolio:")

    picks = []
    for ticker in suggestions:
        if ticker not in st.session_state.portfolio["Ticker"].values:
            try:
                stock = yf.Ticker(ticker)
                time.sleep(1.5)
                hist = stock.history(period="7d")
                current = hist["Close"][-1]
                prev = hist["Close"][0]
                change = ((current - prev) / prev) * 100
                if change > 2:
                    picks.append({"Ticker": ticker, "Change (%)": round(change, 2), "Price": round(current, 2)})
            except:
                continue

    if picks:
        df_picks = pd.DataFrame(picks)
        for _, row in df_picks.iterrows():
            col1, col2, col3 = st.columns([2, 2, 1])
            col1.write(f"**{row['Ticker']}**")
            col2.write(f"{row['Change (%)']}% | ${row['Price']}")
            if col3.button("➕ Add", key=f"add_{row['Ticker']}"):
                new_row = pd.DataFrame({"Ticker": [row["Ticker"]], "Shares": [1]})
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                st.success(f"{row['Ticker']} added with 1 share")
                st.experimental_rerun()
    else:
        st.info("No standout picks this week.")

# ============================
# ℹ️ How It Works Tab
# ============================
with tabs[2]:
    st.title("ℹ️ How It Works")

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
