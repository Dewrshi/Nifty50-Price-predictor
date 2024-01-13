"""This Script will download the nifty50 data of past 10 years
in the form of csv file to train models"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define the symbol for Nifty50
symbol = '^NSEI'

# Calculate the start date (10 years ago from today)
start_date = datetime.now() - timedelta(days=365 * 10)

# Download Nifty50 historical data
nifty_data = yf.download(symbol, start=start_date, end=datetime.now())

# Save the data to a CSV file
nifty_data.to_csv('nifty50_data.csv')
print("Download was success and the the head of your downloaded data is below")

# Display the first few rows of the data
print(nifty_data.head())
