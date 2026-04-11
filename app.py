import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from scipy.stats import skew, kurtosis, norm, probplot, jarque_bera

st.set_page_config(page_title="Stock App", layout="wide")

st.title("Stock Analysis App")

st.sidebar.header("Inputs")

default_start = date.today() - timedelta(days=365*2)
default_end = date.today()

ticker_input = st.sidebar.text_input(
    "Enter tickers (2–5, comma separated)",
    value="AAPL,MSFT"
)

start_date = st.sidebar.date_input("Start date", value=default_start, min_value=date(1970, 1, 1), max_value=date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End date", value=date.today(), min_value=date(1970, 1, 1), max_value=date.today() - timedelta(days=365))

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
        st.error(f"Error downloading data: {e}")
        st.stop()

if data.empty:
    st.error("No data returned. Check your ticker symbols and try again.")
    st.stop()

bad = [t for t in tickers if t not in data.columns or data[t].dropna().empty]
if bad:
    st.error(f"Could not find data for: {', '.join(bad)}. Please check these tickers.")
    st.stop()

returns = data.pct_change().dropna()
stock_only_cols = [col for col in returns.columns if col != "^GSPC"]

tab1, tab2, tab3, tab4 = st.tabs([
    "Price & Returns",
    "Risk & Distribution",
    "Correlation",
    "About"
])

with tab1:
    st.write("### Price Data")
    st.line_chart(data)

    st.write("### Daily Returns")
    st.line_chart(returns)

    stats = pd.DataFrame({
        "Annual Return": returns.mean() * 252,
        "Volatility": returns.std() * np.sqrt(252),
        "Skewness": returns.apply(skew),
        "Kurtosis": returns.apply(kurtosis),
        "Min Return": returns.min(),
        "Max Return": returns.max()
    })

    st.write("### Summary Statistics")
    st.dataframe(stats)

    st.write("### Growth of $10,000 Investment")

    equal_weight_returns = returns[stock_only_cols].mean(axis=1)
    wealth_df = pd.DataFrame(index=returns.index)
    wealth_df["Equal Weight Portfolio"] = (1 + equal_weight_returns).cumprod() * 10000
    for col in returns.columns:
        wealth_df[col] = (1 + returns[col]).cumprod() * 10000

    st.line_chart(wealth_df)

with tab2:
    st.write("### Rolling Volatility")

    window = st.selectbox("Select rolling window (days)", [30, 60, 90], index=1)
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    st.line_chart(rolling_vol)

    st.write("### Distribution Analysis")

    selected_stock = st.selectbox("Select a stock for distribution analysis", stock_only_cols)
    selected_returns = returns[selected_stock].dropna()

    dist_tab1, dist_tab2 = st.tabs(["Histogram", "Q-Q Plot"])

    with dist_tab1:
        mu, sigma = norm.fit(selected_returns)
        x_vals = np.linspace(selected_returns.min(), selected_returns.max(), 200)
        y_vals = norm.pdf(x_vals, mu, sigma)

        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=selected_returns, histnorm="probability density", name="Daily Returns"))
        hist_fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Fitted Normal Curve"))
        hist_fig.update_layout(title=f"Histogram: {selected_stock}", xaxis_title="Daily Return", yaxis_title="Density")
        st.plotly_chart(hist_fig, use_container_width=True)

    with dist_tab2:
        qq = probplot(selected_returns, dist="norm")
        theoretical = qq[0][0]
        ordered = qq[0][1]
        slope = qq[1][0]
        intercept = qq[1][1]

        qq_fig = go.Figure()
        qq_fig.add_trace(go.Scatter(x=theoretical, y=ordered, mode="markers", name="Sample Quantiles"))
        qq_fig.add_trace(go.Scatter(x=theoretical, y=slope * theoretical + intercept, mode="lines", name="Reference Line"))
        qq_fig.update_layout(title=f"Q-Q Plot: {selected_stock}", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        st.plotly_chart(qq_fig, use_container_width=True)

    st.write("### Jarque-Bera Normality Test")
    jb_stat, jb_p = jarque_bera(selected_returns)
    st.write(f"Test Statistic: {jb_stat:.4f}")
    st.write(f"P-value: {jb_p:.4f}")
    if jb_p < 0.05:
        st.error("Rejects normality (p < 0.05)")
    else:
        st.success("Fails to reject normality (p >= 0.05)")

    st.write("### Box Plot of Daily Returns")
    box_fig = px.box(returns[stock_only_cols], title="Box Plot of Daily Returns", labels={"value": "Daily Return", "variable": "Ticker"})
    st.plotly_chart(box_fig, use_container_width=True)

with tab3:
    st.write("### Correlation Heatmap")

    corr_matrix = returns.corr()
    heatmap_fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Heatmap of Daily Returns")
    heatmap_fig.update_layout(xaxis_title="Ticker", yaxis_title="Ticker")
    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.write("### Scatter Plot of Daily Returns")
    stock_a = st.selectbox("Select Stock A", stock_only_cols, index=0, key="scatter_a")
    stock_b = st.selectbox("Select Stock B", stock_only_cols, index=min(1, len(stock_only_cols)-1), key="scatter_b")

    if stock_a == stock_b:
        st.warning("Please select two different stocks.")
    else:
        scatter_fig = px.scatter(returns, x=stock_a, y=stock_b, title=f"Scatter Plot: {stock_a} vs {stock_b}", labels={stock_a: f"{stock_a} Daily Return", stock_b: f"{stock_b} Daily Return"})
        st.plotly_chart(scatter_fig, use_container_width=True)

    st.write("### Rolling Correlation")
    corr_window = st.selectbox("Select rolling window (days)", [30, 60, 90], index=1, key="corr_window")

    if stock_a != stock_b:
        rolling_corr = returns[stock_a].rolling(corr_window).corr(returns[stock_b])
        rolling_corr_fig = px.line(rolling_corr, title=f"{corr_window}-Day Rolling Correlation: {stock_a} vs {stock_b}", labels={"value": "Correlation", "index": "Date"})
        st.plotly_chart(rolling_corr_fig, use_container_width=True)

    st.write("### Two-Asset Portfolio Explorer")
    st.info(
        "Combining two stocks can produce a portfolio with lower volatility than either stock "
        "individually. This diversification effect is strongest when the two stocks have low "
        "or negative correlation. The curve below shows portfolio volatility across all weight combinations."
    )

    portfolio_a = st.selectbox("Select Portfolio Stock A", stock_only_cols, index=0, key="portfolio_a")
    portfolio_b = st.selectbox("Select Portfolio Stock B", stock_only_cols, index=min(1, len(stock_only_cols)-1), key="portfolio_b")

    if portfolio_a == portfolio_b:
        st.warning("Please select two different stocks for the portfolio explorer.")
    else:
        weight_a = st.slider(
            f"Weight on {portfolio_a} (remainder goes to {portfolio_b})",
            min_value=0, max_value=100, value=50, step=1, format="%d%%"
        ) / 100.0
        weight_b = 1 - weight_a

        mean_returns = returns[[portfolio_a, portfolio_b]].mean() * 252
        cov_matrix = returns[[portfolio_a, portfolio_b]].cov() * 252
        weights = np.array([weight_a, weight_b])
        portfolio_return = np.dot(weights, mean_returns.values)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))

        col1, col2 = st.columns(2)
        col1.metric("Annualized Return", f"{portfolio_return:.2%}")
        col2.metric("Annualized Volatility", f"{portfolio_vol:.2%}")

        weight_range = np.linspace(0, 1, 101)
        portfolio_vols = []
        for w in weight_range:
            w_vec = np.array([w, 1 - w])
            p_vol = np.sqrt(np.dot(w_vec.T, np.dot(cov_matrix.values, w_vec)))
            portfolio_vols.append(p_vol)

        frontier_df = pd.DataFrame({f"Weight in {portfolio_a}": weight_range, "Annualized Volatility": portfolio_vols})
        frontier_fig = px.line(frontier_df, x=f"Weight in {portfolio_a}", y="Annualized Volatility", title=f"Portfolio Volatility Curve: {portfolio_a} vs {portfolio_b}")
        frontier_fig.add_scatter(x=[weight_a], y=[portfolio_vol], mode="markers", marker=dict(size=12, color="red"), name="Current Weight")
        frontier_fig.update_layout(xaxis_title=f"Weight in {portfolio_a} (0=100% {portfolio_b}, 1=100% {portfolio_a})", yaxis_title="Annualized Volatility")
        st.plotly_chart(frontier_fig, use_container_width=True)

with tab4:
    st.write("### About / Methodology")
    st.write("""
    This application compares multiple stocks using historical adjusted closing prices from Yahoo Finance via yfinance.

    **Key assumptions:**
    - Returns are simple arithmetic returns computed using pct_change().
    - Annualized return = mean daily return × 252.
    - Annualized volatility = daily standard deviation × √252.
    - Cumulative wealth index uses (1 + r).cumprod() starting at $10,000.
    - Equal-weight portfolio return = average of daily returns across all selected stocks.
    - Two-asset portfolio uses classical mean-variance formulas with annualized covariance.
    - Rolling calculations produce NaN for the first (window - 1) observations — this is expected.
    - Data source: Yahoo Finance (yfinance). Benchmark: S&P 500 (^GSPC).
    - Data is cached for 1 hour using @st.cache_data.
    """)