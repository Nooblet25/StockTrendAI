import yfinance as yf
import pandas as pd

# Test with a simple API call
print("Fetching AAPL data...")
stock = yf.Ticker("AAPL")
data = stock.history(period="5d")

print("\nLast 5 days of AAPL data:")
print(data) 