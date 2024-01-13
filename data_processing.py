"""This script will show the average, buy and sell
values on a graph."""

import pandas as pd
import matplotlib.pyplot as plt


# Load the data from the CSV file
nifty_data = pd.read_csv('nifty50_data.csv', index_col='Date', parse_dates=True)

# Perform data processing (e.g., calculate moving averages)
nifty_data['MA_50'] = nifty_data['Close'].rolling(window=50).mean()
nifty_data['MA_200'] = nifty_data['Close'].rolling(window=200).mean()

# Generate buy (1), sell (-1), or hold (0) signals based on moving average crossover
nifty_data['Signal'] = 0  # Default to hold
nifty_data['Signal'][nifty_data['MA_50'] > nifty_data['MA_200']] = 1  # Buy signal
nifty_data['Signal'][nifty_data['MA_50'] < nifty_data['MA_200']] = -1  # Sell signal

# Drop NaN values introduced by the moving averages
nifty_data = nifty_data.dropna()

# Backtesting: Calculate strategy returns
nifty_data['Daily_Return'] = nifty_data['Close'].pct_change()
nifty_data['Strategy_Return'] = nifty_data['Daily_Return'] * nifty_data['Signal'].shift(1)

# Calculate cumulative returns
nifty_data['Cumulative_Return'] = (1 + nifty_data['Strategy_Return']).cumprod()

# Print the results
print(nifty_data[['Signal', 'Strategy_Return', 'Cumulative_Return']].head())

# Visualize the strategy signals and cumulative returns (optional)

plt.figure(figsize=(10, 6))
nifty_data['Close'].plot(label='Nifty50 Close Price')
plt.scatter(nifty_data.index[nifty_data['Signal'] == 1], nifty_data['Close'][nifty_data['Signal'] == 1], label='Buy Signal', marker='^', color='g')
plt.scatter(nifty_data.index[nifty_data['Signal'] == -1], nifty_data['Close'][nifty_data['Signal'] == -1], label='Sell Signal', marker='v', color='r')
plt.legend()
plt.show()
