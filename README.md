# Intelligent Stock Portfolio Tracker

This Streamlit app allows you to upload your stock portfolio (ticker + shares), then:
- Retrieves historical price data from Yahoo Finance
- Calculates total current value of each position
- Flags BUY signals using 50-day vs 200-day moving average crossover
- Projects portfolio value over 1, 3, and 5 years assuming CAGR
- Visualizes price trends with overlays

---

## 🧾 How to Use

1. Clone this repo or download the files
2. Install the requirements:

```bash
pip install streamlit yfinance matplotlib pandas
