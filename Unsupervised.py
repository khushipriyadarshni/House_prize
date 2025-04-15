import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Data Collection
data = pd.read_csv(r"C:\Users\khush\OneDrive\Desktop\Projects\House.prize\delhi_house_prices.csv")

# Data Preprocessing
# Handle missing values (if any)
data.fillna(data.mean(), inplace=True)

# Normalize numerical features
features = ['latitude', 'longitude', 'size', 'age', 'bedrooms', 'bathrooms', 'garage', 'pool']
X = data[features]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

#  Clustering
# Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 6))
sns.lineplot(x=range(1, 11), y=inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# From the elbow plot, assume the optimal number of clusters is 4
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_features)

# Predictive Modeling
# Train a predictive model for each cluster
models = {}
for cluster in data['cluster'].unique():
    cluster_data = data[data['cluster'] == cluster]
    X_train, X_test, y_train, y_test = train_test_split(cluster_data[features], cluster_data['price'], test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Cluster {cluster} MAE: {mae}')
    
    models[cluster] = model

# Prediction on New Data
# Predict the price of a new house
new_house = pd.DataFrame([{
    'latitude': 28.6139, 'longitude': 77.2090, 'size': 1500, 'age': 5, 
    'bedrooms': 3, 'bathrooms': 2, 'garage': 1, 'pool': 0
}])
scaled_new_house = scaler.transform(new_house)
new_cluster = kmeans.predict(scaled_new_house)[0]
predicted_price = models[new_cluster].predict(new_house)[0]
print(f'Predicted price: {predicted_price}')