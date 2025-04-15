import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Data Collection
data = pd.read_csv(r"C:\Users\khush\OneDrive\Desktop\Projects\House.prize\delhi_house_prices.csv")

# Data Preprocessing
# Handle missing values
data.fillna(data.mean(), inplace=True)

# Normalize numerical features
features = data[['latitude', 'longitude', 'size', 'age', 'bedrooms', 'bathrooms', 'garage', 'pool']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Feature Selection
# Here we are using all features for prediction

#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, data['price'], test_size=0.2, random_state=42)

#  Model Training
model = LinearRegression()
model.fit(X_train, y_train)

#  Model Evaluation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics
mae_train = mean_absolute_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f'Training MAE: {mae_train}')
print(f'Training MSE: {mse_train}')
print(f'Training R2: {r2_train}')

print(f'Test MAE: {mae_test}')
print(f'Test MSE: {mse_test}')
print(f'Test R2: {r2_test}')

# Plotting true vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title('True Prices vs Predicted Prices')
plt.show()

#  Prediction on New Data
# Predict the price of a new house
new_house = pd.DataFrame([{
    'latitude': 28.6139, 'longitude': 77.2090, 'size': 1500, 'age': 5, 
    'bedrooms': 3, 'bathrooms': 2, 'garage': 1, 'pool': 0
}])
scaled_new_house = scaler.transform(new_house)
predicted_price = model.predict(scaled_new_house)[0]
print(f'Predicted price: {predicted_price}')