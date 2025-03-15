import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import messagebox


data = pd.read_csv("data.csv")

#Data Cleaning
data['PURPOSE'] = data['PURPOSE'].fillna('Unknown')

#Date Conversion
data['START_DATE'] = pd.to_datetime(data['START_DATE'], errors='coerce', dayfirst=False)

#Checking for invalid dates after conversion (rows with NaT)
invalid_dates = data[data['START_DATE'].isna()]
if not invalid_dates.empty:
    print("Rows with invalid dates:")
    print(invalid_dates)
else:
    print("No invalid dates found.")

#Checking for any other missing values in numerical or categorical columns
missing_data = data.isnull().sum()
print("\nMissing values in the dataset:")
print(missing_data)

#Filling missing values in other numerical columns, like 'MILES'
data['MILES'] = data['MILES'].fillna(data['MILES'].mean())


data['CATEGORY'] = data['CATEGORY'].fillna('Unknown')
data['START'] = data['START'].fillna('Unknown')
data['END_DATE'] = data['END_DATE'].fillna('Unknown')
data['STOP'] = data['STOP'].fillna('Unknown')

#Clustering
print("\nPerforming KMeans clustering on MILES column...")
scaler = StandardScaler()
data[['MILES']] = scaler.fit_transform(data[['MILES']])

kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['MILES']])

print("\nData with Cluster Labels:")
print(data[['MILES', 'Cluster']].head())


if 'START_DATE' in data.columns:
    filtered_data = data[data['START_DATE'] > '2020-01-01']
    print("\nFiltered Data (Trips started after 2020):")
    print(filtered_data.head())
else:
    print("\n'COLUMN START_DATE' not found. Skipping filtering step.")


duplicates = data.duplicated().sum()
if duplicates > 0:
    print(f"\nFound {duplicates} duplicate rows. Removing duplicates...")
    data = data.drop_duplicates()

#Normalized
print("\nData after normalization:")
print(data[['MILES']].head())

# Visuals
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data.index, y=data['MILES'], hue=data['Cluster'], palette='Set2', s=100)
plt.title('Cluster Visualization based on MILES')
plt.xlabel('Index')
plt.ylabel('Normalized MILES')
plt.legend(title='Cluster')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', data=data, palette='Set2')
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

#WCSS Elbow
wcss = []
for i in range(1, 11):
    kmeans_temp = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans_temp.fit(data[['MILES']])
    wcss.append(kmeans_temp.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#Overfitting and Underfitting using Train-Test Split
X = data[['MILES']]
y = data['Cluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_train = linear_model.predict(X_train)
y_pred_test = linear_model.predict(X_test)

train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f"\nTrain MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
if train_mse < test_mse:
    print("\nThe model may be overfitting.")
elif train_mse > test_mse:
    print("\nThe model may be underfitting.")
else:
    print("\nThe model seems to be well-fitted.")

# Calculate accuracy
y_pred_test_rounded = np.round(y_pred_test)
accuracy = accuracy_score(y_test, y_pred_test_rounded)
print(f"\nModel Accuracy 7{accuracy * 100:}%")


print("\nPerforming Hyperparameter Tuning for KMeans using GridSearchCV...")
param_grid = {
    'n_clusters': [2, 3, 4, 5, 6],
    'max_iter': [200, 300, 400],
    'init': ['k-means++', 'random']
}
grid_search = GridSearchCV(KMeans(random_state=42), param_grid, cv=3)
grid_search.fit(data[['MILES']])

print(f"Best parameters from GridSearchCV: {grid_search.best_params_}")

#final Cluster Distribution
print("\nFinal Cluster Distribution:")
print(data['Cluster'].value_counts())


data.to_csv("processed_data.csv", index=False)
print("\nData saved to 'processed_data.csv'.")

#Final Visualization - PCA for Dimensionality Reduction
pca = PCA(n_components=1)
pca_components = pca.fit_transform(data[['MILES']])
pca_components_flat = pca_components.flatten()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_components_flat, y=np.zeros_like(pca_components_flat), hue=data['Cluster'], palette='Set2')
plt.title('PCA of Clusters (with 1 Component)')
plt.xlabel('PCA Component 1')
plt.ylabel('Zero')
plt.legend(title='Cluster')
plt.show()


def predict_cluster():
    """Predicts the cluster based on user input."""
    try:
        miles = float(entry_miles.get()) if entry_miles.get().strip() else 0
        purpose = entry_purpose.get().strip() if entry_purpose.get().strip() else "Unknown"
        category = entry_category.get().strip() if entry_category.get().strip() else "Unknown"
        start = entry_start.get().strip() if entry_start.get().strip() else "Unknown"

        # Prepare input with correct feature names
        input_data = pd.DataFrame(columns=data.columns[:-1])  # Use features from training data
        input_data.loc[0] = 0  # Initialize with zeros

        # Fill in numerical data
        input_data['MILES'] = miles

        # Handle categorical features dynamically
        for col in input_data.columns:
            if col.startswith("PURPOSE_") and f'PURPOSE_{purpose}' == col:
                input_data[col] = 1
            if col.startswith("CATEGORY_") and f'CATEGORY_{category}' == col:
                input_data[col] = 1
            if col.startswith("START_") and f'START_{start}' == col:
                input_data[col] = 1

        # Scaling numerical features
        input_data['MILES'] = scaler.transform(input_data[['MILES']])

        # Predicting the cluster
        predicted_cluster = kmeans.predict(input_data[['MILES']])[0]
        messagebox.showinfo("Prediction", f"The data belongs to Cluster {predicted_cluster}")
    except ValueError:
        messagebox.showerror("Error", "Invalid numerical input. Please enter valid numbers.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

#GUI
root = tk.Tk()
root.title("Cluster Prediction")
root.geometry("400x400")

tk.Label(root, text="Enter the values for prediction:").pack(pady=10)

tk.Label(root, text="MILES (e.g., 150):").pack()
entry_miles = tk.Entry(root)
entry_miles.pack(pady=5)

tk.Label(root, text="PURPOSE:").pack()
entry_purpose = tk.Entry(root)
entry_purpose.pack(pady=5)

tk.Label(root, text="CATEGORY:").pack()
entry_category = tk.Entry(root)
entry_category.pack(pady=5)

tk.Label(root, text="START:").pack()
entry_start = tk.Entry(root)
entry_start.pack(pady=5)

tk.Button(root, text="Predict Cluster", command=predict_cluster).pack(pady=10)
root.mainloop()