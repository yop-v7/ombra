# app.py
import streamlit as st
import numpy as np
import datetime
import matplotlib.pyplot as plt
from ombra import get_polygon_daily_data, weekly_price_prediction
from dotenv import load_dotenv
import os

# Load environment variables (Polygon API Key)
load_dotenv()

st.set_page_config(page_title="OMBRA Investment Advisor", layout="wide")
st.title("OMBRA Investment Advisor")
st.markdown("**Publisher:** Tekk Systems")

# Sidebar ticker input and date selection
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol", value="AAPL")
st.sidebar.markdown("### Select Date Range for Historical Data")
today = datetime.date.today()
default_start = today - datetime.timedelta(days=90)
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=today)

# Function: Black-Scholes based forecast
def forecast_black_scholes(prices, days=7):
    # Use historical daily log returns to compute the drift
    log_returns = np.diff(np.log(prices))
    mean_daily_return = np.mean(log_returns)
    current_price = prices[-1]
    # Forecast using exponential growth over days (T = days/252 trading days per year)
    forecast = current_price * np.exp(mean_daily_return * days)
    return forecast

# Function: Regression based forecast (linear trend)
def forecast_regression(prices, days=7):
    n = len(prices)
    x = np.arange(n)
    # Fit a simple linear regression
    slope, intercept = np.polyfit(x, prices, 1)
    forecast = slope * (n - 1 + days) + intercept
    return forecast

# Function: Time Series forecast (naïve method using average daily change)
def forecast_time_series(prices, days=7):
    daily_change = np.mean(np.diff(prices))
    current_price = prices[-1]
    forecast = current_price + daily_change * days
    return forecast

# Function: Investment recommendation based on percentage change
def get_recommendation(current, forecast):
    pct_change = (forecast - current) / current * 100
    if pct_change >= 5:
        rec = "Strong Buy"
    elif pct_change > 0:
        rec = "Weak Buy"
    elif pct_change <= -5:
        rec = "Strong Sell"
    else:
        rec = "Weak Sell"
    return pct_change, rec

# Main block: fetch data and generate forecasts
if st.button("Run Analysis"):
    try:
        # Format dates for Polygon API (YYYY-MM-DD)
        from_date = start_date.strftime("%Y-%m-%d")
        to_date = end_date.strftime("%Y-%m-%d")

        # Get historical prices using ombra's function
        prices = get_polygon_daily_data(ticker.upper(), from_date, to_date)
        if prices.size == 0:
            st.error("No price data found. Please check the ticker or date range.")
        else:
            current_price = prices[-1]
            st.subheader(f"Current Price of {ticker.upper()}: ${current_price:.2f}")

            # Forecasts: each method produces a 7-day forecast price
            ombra_forecasts = weekly_price_prediction(prices, days_ahead=7)
            forecast_ombra = ombra_forecasts[-1]

            forecast_bs = forecast_black_scholes(prices, days=7)
            forecast_regr = forecast_regression(prices, days=7)
            forecast_ts = forecast_time_series(prices, days=7)

            # Create a dictionary of forecasts and recommendations
            methods = {
                "OMBRA": forecast_ombra,
                "Black-Scholes": forecast_bs,
                "Regression": forecast_regr,
                "Time Series": forecast_ts,
            }

            st.markdown("### Forecast Summary for 7 Days Ahead")
            for method, forecast in methods.items():
                pct_change, recommendation = get_recommendation(current_price, forecast)
                st.write(
                    f"**{method}:** Predicted Price = ${forecast:.2f} | Change = {pct_change:.2f}% | Recommendation: {recommendation}"
                )

            # Plot historical prices and forecast lines
            fig, ax = plt.subplots(figsize=(10, 5))
            n = len(prices)
            days_extended = n + 7

            # Plot historical prices
            ax.plot(np.arange(n), prices, label="Historical Prices")

            # For each method, plot a dashed line from last historical point to forecast at day n-1+7
            for method, forecast in methods.items():
                ax.plot([n - 1, days_extended - 1], [current_price, forecast], linestyle="--", marker="o", label=f"{method} Forecast")

            ax.set_xlabel("Time (Days)")
            ax.set_ylabel("Price")
            ax.set_title(f"{ticker.upper()} Price Forecast")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("----")
st.markdown("**Note:** Ensure you have a valid `POLYGON_API_KEY` set as an environment variable.")
