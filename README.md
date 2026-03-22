# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM :

## GIVEN DATA:

```

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os

# --- Independent Path Finder ---
file_path = None
for path in ['/aapl_master_enriched.csv', '/content/aapl_master_enriched.csv', 'aapl_master_enriched.csv']:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    print("Error: File not found. Please ensure 'aapl_master_enriched.csv' is uploaded.")
else:
    # Load and explore
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    data = df['close'].tail(500) # Using latest 500 days

    print("--- OUTPUT: GIVEN DATA ---")
    print(data.head())

    # ADF Test for Stationarity
    result = adfuller(data)
    print(f'\nADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(data, color='#1f77b4')
    plt.title('Apple Stock Close Price: Raw Given Data')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.show()

```

## PACF - ACF:

```

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

# --- Independent Path Finder ---
file_path = None
for path in ['/aapl_master_enriched.csv', '/content/aapl_master_enriched.csv', 'aapl_master_enriched.csv']:
    if os.path.exists(path):
        file_path = path
        break

if file_path:
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data = df['close'].tail(500)

    print("--- OUTPUT: PACF - ACF ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    plot_acf(data, ax=ax1, lags=40, title='Apple Stock: Autocorrelation (ACF)')
    plot_pacf(data, ax=ax2, lags=40, title='Apple Stock: Partial Autocorrelation (PACF)')

    plt.tight_layout()
    plt.show()

```

## FINIAL PREDICTION:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import os

# --- Independent Path Finder ---
file_path = None
for path in ['/aapl_master_enriched.csv', '/content/aapl_master_enriched.csv', 'aapl_master_enriched.csv']:
    if os.path.exists(path):
        file_path = path
        break

if file_path:
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data_values = df['close'].tail(500).values
    train_size = int(len(data_values) * 0.8)
    train, test = data_values[:train_size], data_values[train_size:]

    # Fit and Predict
    model = AutoReg(train, lags=13).fit()
    predictions = model.predict(start=len(train), end=len(train) + len(test) - 1)

    # Evaluation
    mse = mean_squared_error(test, predictions)

    print("--- OUTPUT: FINAL PREDICTION ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")

    # Final Comparison Plot
    plt.figure(figsize=(12, 6))
    plt.plot(test, label='Actual Apple Close Price', color='black', linewidth=1.5)
    plt.plot(predictions, label='AR(13) Model Prediction', color='#d62728', linestyle='--', linewidth=2)
    plt.title('Final Prediction: Actual vs AR Model (Apple Stock)')
    plt.xlabel('Days into Test Set')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

```

## PREDICTION:

```

import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import os

# --- Independent Path Finder ---
file_path = None
for path in ['/aapl_master_enriched.csv', '/content/aapl_master_enriched.csv', 'aapl_master_enriched.csv']:
    if os.path.exists(path):
        file_path = path
        break

if file_path:
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data_values = df['close'].tail(500).values

    # 80/20 Train-Test Split
    train_size = int(len(data_values) * 0.8)
    train, test = data_values[:train_size], data_values[train_size:]

    # Fit AutoRegressive Model (13 lags)
    model = AutoReg(train, lags=13).fit()

    # Generate Predictions
    predictions = model.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

    print("--- OUTPUT: PREDICTION ---")
    print(f"Total Data: {len(data_values)} | Training: {len(train)} | Testing: {len(test)}")
    print("\nFirst 5 Predicted Prices for Apple Stock:")
    print(predictions[:5])

```
   
### OUTPUT:

<img width="827" height="549" alt="image" src="https://github.com/user-attachments/assets/04748dc4-0cff-473f-99c8-baba5d77a833" />
<img width="1398" height="469" alt="image" src="https://github.com/user-attachments/assets/5d8a3172-8cdf-49c8-8ace-37a57931487f" />
<img width="516" height="79" alt="image" src="https://github.com/user-attachments/assets/788ef84d-06f9-4b73-bc80-35e489573420" />
<img width="952" height="546" alt="image" src="https://github.com/user-attachments/assets/6a8e82f4-3b01-4bcb-ba46-e3129bedc242" />

### RESULT:
Thus we have successfully implemented the auto regression function using python.
