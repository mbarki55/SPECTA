import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Define the dataset (replace this with actual CSV loading if available)
data = {
    'Center': ['Nabil', 'Said', 'Mounir', 'Samia', 'Amira', 'Farah', 'Imane', 'Hafesa', 'Noura', 'Khalid', 
               'Salim', 'Ghita', 'Najlae', 'Oumaima', 'Hamza', 'Simo', 'Narjis', 'Houda', 'Hassan', 'Imad'],
    'KPI': [0.66, 0.28, 0.47, 0.88, 0.26, 0.56, 0.77, 0.59, 0.68, 0.78, 0.59, 0.25, 0.49, 0.33, 0.65, 0.22, 0.32, 0.51, 0.91, 0.74]
}
df = pd.DataFrame(data)

# Choose a target KPI, for example, 0.7 to find similar centers
target_kpi = [[0.7]]

# Fit KNN model
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(df[['KPI']])

# Find the closest centers
distances, indices = knn.kneighbors(target_kpi)

# Display the closest centers
closest_centers = df.iloc[indices[0]]
print("Nearest centers to target KPI score:\n", closest_centers)
