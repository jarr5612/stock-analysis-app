import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from scipy.stats import skew, kurtosis, norm, probplot, jarque_bera

st.set_page_config(page_title="Stock App", layout="wide")

st.write("Hello World")

st.title("Stock Analysis App")

st.sidebar.header("Inputs")

default_start = date.today() - timedelta(days=365*2)
default_end = date.today()

ticker_input = st.sidebar.text_input(
    "Enter tickers (2–5, comma separated)",
    value="AAPL,MSFT"
)

start_date = st.sidebar.date_input("Start date", default_start)
end_date = st.sidebar.date_input("End date", default_end)

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

if len(tickers) < 2 or len(tickers) > 5:
    st.error("Enter between 2 and 5 tickers.")
    st.stop()

if (end_date - start_date).days < 365:
    st.error("Date range must be at least 1 year. Please adjust your dates.")
    st.stop()

@st.cache_data(ttl=3600)
def get_data(tickers, start, end):
    all_tickers = list(tickers) + ["^GSPC"]
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
    return prices

with st.spinner("Loading data..."):
    try:
        data = get_data(tuple(tickers), start_date, end_date)
    except Exception as e:
        st.error