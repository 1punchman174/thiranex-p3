import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv("Walmart.csv")

print("Dataset Preview:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

X = df[['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
        'CPI', 'Unemployment', 'Year', 'Month', 'Day']]

y = df['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

results = pd.DataFrame({
    'Actual Sales': y_test,
    'Predicted Sales': predictions
})

print("\nActual vs Predicted:")
print(results.head())

plt.figure(figsize=(10, 5))

plt.plot(
    y_test.values[:50],
    label='Actual Sales'
)

plt.plot(
    predictions[:50],
    label='Predicted Sales'
)

plt.title("Actual vs Predicted Weekly Sales")
plt.xlabel("Records")
plt.ylabel("Sales")
plt.legend()

plt.savefig("sales_prediction.png")

plt.show()

future_data = pd.DataFrame({
    'Store': [1],
    'Holiday_Flag': [0],
    'Temperature': [75],
    'Fuel_Price': [3.5],
    'CPI': [220],
    'Unemployment': [7],
    'Year': [2026],
    'Month': [6],
    'Day': [15]
})

future_sales = model.predict(future_data)

print("\nFuture Sales Prediction:")
print("Predicted Weekly Sales:", future_sales[0])