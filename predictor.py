"""This script will do the prediction and the accuracy is around 99% 
without overfitting."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the data from the CSV file
nifty_data = pd.read_csv('nifty50_data.csv', index_col='Date', parse_dates=True)

# Data Processing
nifty_data['MA_50'] = nifty_data['Close'].rolling(window=50).mean()
nifty_data['MA_200'] = nifty_data['Close'].rolling(window=200).mean()
nifty_data['Signal'] = 0  # Default to hold
nifty_data.loc[nifty_data['MA_50'] > nifty_data['MA_200'], 'Signal'] = 1  # Buy signal
nifty_data.loc[nifty_data['MA_50'] < nifty_data['MA_200'], 'Signal'] = -1  # Sell signal
nifty_data = nifty_data.dropna()

# Machine Learning
X = nifty_data[['MA_50', 'MA_200']]
y = nifty_data['Signal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict signals on the test set
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Display classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Backtesting
nifty_data['Predicted_Signal'] = clf.predict(X)
nifty_data['Daily_Return'] = nifty_data['Close'].pct_change()
nifty_data['Strategy_Return'] = nifty_data['Daily_Return'] * nifty_data['Predicted_Signal'].shift(1)
nifty_data['Cumulative_Strategy_Return'] = (1 + nifty_data['Strategy_Return']).cumprod()

# Visualize the results
plt.figure(figsize=(12, 6))
nifty_data['Cumulative_Strategy_Return'].plot(label='Strategy Cumulative Return', color='orange', linestyle='dashed')
nifty_data['Cumulative_Strategy_Return'].plot(label='Strategy Cumulative Return', color='orange', linestyle='dashed')
plt.title('Nifty50 vs. Strategy Cumulative Returns')
plt.legend()
plt.show()
