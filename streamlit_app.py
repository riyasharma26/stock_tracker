import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from sklearn.linear_model import LinearRegression
import io
from PIL import Image
import base64

# ---------------- Setup ----------------
st.set_page_config(page_title="Intelligent Stock Portfolio Tracker", layout="wide")
st.title("Intelligent Stock Portfolio Tracker")

# Initialize session state
if "portfolio" not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=["Ticker", "Shares"])

# Upload section
uploaded_file = st.sidebar.file_uploader("Upload Portfolio CSV (Ticker, Shares)", type=["csv"])
if uploaded_file:
    st.session_state.portfolio = pd.read_csv(uploaded_file)

# Manual entry
with st.sidebar.form(key="manual_entry"):
    st.write("Add a stock manually")
    ticker_input = st.text_input("Ticker")
    shares_input = st.number_input("Shares Owned", min_value=0.01, step=1.0)
    submit = st.form_submit_button("Add Stock")
    if submit and ticker_input:
        new_row = pd.DataFrame({"Ticker": [ticker_input.upper()], "Shares": [shares_input]})
        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
        st.success(f"Added {ticker_input.upper()}")

# Tabs
tab1, tab2 = st.tabs(["📊 Portfolio Tracker", "📈 Weekly Picks"])

# ---------------- TAB 1: Portfolio Tracker ----------------
with tab1:
    st.subheader("Your Current Portfolio")
    portfolio = st.session_state.portfolio
    to_remove = []

    if not portfolio.empty:
        for idx, row in portfolio.iterrows():
            col1, col2, col3 = st.columns([3, 1, 0.2])
            col1.write(f"**{row['Ticker']}** — Shares: {row['Shares']}")
            if col3.button("🗑️", key=f"remove_{idx}"):
                to_remove.append(idx)

        if to_remove:
            st.session_state.portfolio.drop(to_remove, inplace=True)
            st.session_state.portfolio.reset_index(drop=True, inplace=True)
            st.rerun()

        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)
        projections = []

        for index, row in portfolio.iterrows():
            ticker = row["Ticker"]
            shares = row["Shares"]

            try:
                stock = yf.Ticker(ticker)
                time.sleep(1.5)
                hist = stock.history(start=start_date, end=end_date)

                if hist.empty:
                    st.warning(f"No data found for {ticker}")
                    continue

                current_price = hist["Close"][-1]
                avg_50 = hist["Close"].rolling(window=50).mean().iloc[-1]
                avg_200 = hist["Close"].rolling(window=200).mean().iloc[-1]
                total_value = current_price * shares

                # Trendline prediction
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

                projections.append({
                    "Ticker": ticker,
                    "Current Price": round(current_price, 2),
                    "Total Value": round(total_value, 2),
                    "50-Day MA": round(avg_50, 2),
                    "200-Day MA": round(avg_200, 2),
                    "Est. Buy Price": est_buy,
                    "Est. Sell Price": est_sell
                })

                # Price chart
                st.write(f"#### {ticker} - Price Trend")
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

        # Portfolio insights
        if projections:
            proj_df = pd.DataFrame(projections)
            st.subheader("Portfolio Insights")
            st.dataframe(proj_df)

            # Download CSV
            csv = proj_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Insights as CSV",
                data=csv,
                file_name="portfolio_insights.csv",
                mime="text/csv"
            )
    else:
        st.info("Upload a CSV or add stocks manually to begin.")

# ---------------- TAB 2: Weekly Picks ----------------
with tab2:
    st.subheader("This Week’s Growth Picks")
    st.markdown("These stocks are not in your portfolio but show strong momentum and may be good buys:")

    try:
        top_candidates = ["NVDA", "TSLA", "MSFT", "GOOGL", "AMZN"]
        start = datetime.today() - timedelta(days=365)
        end = datetime.today()

        weekly_recs = []

        for ticker in top_candidates:
            if ticker in st.session_state.portfolio["Ticker"].values:
                continue

            stock = yf.Ticker(ticker)
            time.sleep(1.5)
            hist = stock.history(start=start, end=end)

            if hist.empty:
                continue

            current_price = hist["Close"][-1]
            avg_50 = hist["Close"].rolling(window=50).mean().iloc[-1]
            avg_200 = hist["Close"].rolling(window=200).mean().iloc[-1]

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

            weekly_recs.append({
                "Ticker": ticker,
                "Current Price": round(current_price, 2),
                "50-Day MA": round(avg_50, 2),
                "200-Day MA": round(avg_200, 2),
                "Est. Buy Price": est_buy,
                "Est. Sell Price": est_sell
            })

        if weekly_recs:
            df = pd.DataFrame(weekly_recs)
            st.dataframe(df)

            for i, row in df.iterrows():
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"**{row['Ticker']}** — Est. Buy: ${row['Est. Buy Price']}, Est. Sell: ${row['Est. Sell Price']}")
                if col2.button("Add", key=f"add_{row['Ticker']}"):
                    new_row = pd.DataFrame({"Ticker": [row['Ticker']], "Shares": [1]})
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                    st.success(f"Added {row['Ticker']} to portfolio")
        else:
            st.info("All suggested stocks are already in your portfolio!")

    except Exception as e:
        st.error(f"Failed to fetch weekly picks: {e}")
