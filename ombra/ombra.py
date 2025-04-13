import os
import numpy as np
import requests
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from scipy.fftpack import fft
from scipy.signal import find_peaks

# Load API Key
load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")

# Polygon.io API parameters
SYMBOL = "AAPL"
TIMEFRAME = "day"
FROM_DATE = "2025-02-11"
TO_DATE = "2025-04-11"

# Fetch daily stock data from Polygon.io
def get_polygon_daily_data(symbol, from_date, to_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{TIMEFRAME}/{from_date}/{to_date}?apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "results" not in data:
        raise ValueError("Invalid API response. Check API Key or request limits.")

    prices = np.array([d["c"] for d in data["results"]])
    return prices

# Compute volatility (standard deviation of log returns)
def compute_volatility(prices, window=20):
    returns = np.diff(np.log(prices))
    return np.std(returns[-window:])

# Compute moving average as equilibrium price
def moving_average(prices, window=20):
    return np.mean(prices[-window:])

# Risk penalty function
def risk_penalty(Z, Z_trend, sigma, alpha=0.05, beta=0.02):
    return alpha * (Z - Z_trend) ** 2 + beta * sigma

# Probability density function combining cyclical and risk constraints
def constrained_probability_density(Z, A, k, omega_values, phi_values, Z_trend, sigma):
    signal = np.zeros_like(Z)
    t = 1  # single time-step forward
    for omega, phi in zip(omega_values, phi_values):
        signal += A * np.cos(k * Z - omega * t + phi)

    raw_probability = signal**2
    penalty = risk_penalty(Z, Z_trend, sigma)

    P_final = raw_probability * np.exp(-penalty)
    P_final /= np.sum(P_final)
    return P_final

# Fourier Transform for dominant market frequencies
def compute_market_frequencies(prices):
    yf = fft(prices - np.mean(prices))
    power_spectrum = np.abs(yf)**2
    peaks, _ = find_peaks(power_spectrum[1:len(power_spectrum)//2])

    if len(peaks) < 3:
        return [2 * np.pi / len(prices)]

    dominant_freqs = sorted(peaks[np.argsort(power_spectrum[peaks])[-3:]])
    return [2 * np.pi * f / len(prices) for f in dominant_freqs]

# 7-day predictive algorithm
def weekly_price_prediction(prices, days_ahead=7):
    future_predictions = []
    price_series = prices.copy()

    for day in range(days_ahead):
        sigma_t = compute_volatility(price_series)
        Z_trend = moving_average(price_series)
        A = np.exp(-sigma_t)

        returns = np.diff(np.log(price_series))
        k = np.mean(returns[-5:])  # smoother momentum estimate

        omega_values = compute_market_frequencies(price_series)
        phi_values = [np.arctan(np.mean(np.gradient(price_series)[-5:]) / (k + 1e-5)) for _ in omega_values]

        Z_range = np.linspace(price_series[-1] * 0.95, price_series[-1] * 1.05, 1000)
        P_Z = constrained_probability_density(Z_range, A, k, omega_values, phi_values, Z_trend, sigma_t)

        predicted_price = Z_range[np.argmax(P_Z)]
        future_predictions.append(predicted_price)
        price_series = np.append(price_series, predicted_price)

    return future_predictions

# Execution
if __name__ == "__main__":
    try:
        prices = get_polygon_daily_data(SYMBOL, FROM_DATE, TO_DATE)
        predictions = weekly_price_prediction(prices)

        print("Predicted Prices for the Next 7 Days:")
        for idx, pred in enumerate(predictions, 1):
            print(f"Day {idx}: ${pred:.2f}")

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(prices)), prices, label="Historical Prices", color="green")
        plt.plot(range(len(prices), len(prices)+7), predictions, "--", label="Predicted Prices", color="blue")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.title(f"7-Day Market Predictions for {SYMBOL}")
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")