import pandas as pd
from datetime import datetime, timedelta

def fetch_live_data(ticker, start="2010-01-01"):
    """
    Fetch stock data from local CSV files
    """
    print(f"Fetching data for {ticker}...")
    
    # Clean up the ticker symbol
    ticker = ticker.strip().upper()
    
    try:
        # Try to read from the local CSV file
        csv_path = f"data/{ticker}.csv"
        df = pd.read_csv(csv_path)
        
        # Convert date column and start date to UTC
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        start_date = pd.to_datetime(start).tz_localize(None)
        
        # Filter by start date
        df = df[df['Date'] >= start_date]
        
        if df.empty:
            print(f"No data available for {ticker} after {start}")
            return pd.DataFrame()
            
        # Ensure we have all required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            print(f"Missing required columns in {csv_path}. Available columns: {df.columns.tolist()}")
            return pd.DataFrame()

        df = df[required_columns]
        print(f"Successfully loaded {len(df)} rows of data for {ticker}")
        print("\nFirst few rows:")
        print(df.head())
        print("\nLast few rows:")
        print(df.tail())
        
        # Print date range
        print(f"\nDate range: {df['Date'].min()} to {df['Date'].max()}")
        return df

    except FileNotFoundError:
        print(f"No data file found for {ticker}. Please make sure the CSV file exists in the data directory.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data for {ticker}: {str(e)}")
        return pd.DataFrame()

# Test with TSLA using a date range from 2023
print("\nTesting TSLA data fetch...")
start_date = "2023-01-01"
df = fetch_live_data("TSLA", start=start_date)

# Also test with AAPL for comparison
if df.empty:
    print("\nTrying with a different ticker (AAPL) for comparison...")
    df = fetch_live_data("AAPL", start=start_date) 