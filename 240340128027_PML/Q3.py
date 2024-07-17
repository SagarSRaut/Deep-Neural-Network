import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

# Load the Iris dataset
df = load_iris()
iris = df.data

 
scaler = StandardScaler()
df_scaled = scaler.fit_transform(iris)

# Define the range of clusters to evaluate
Ks = [2, 3, 4, 5]
scores = []

# Iterate over different numbers of clusters
for k in Ks:
    # Fit KMeans clustering model
    clust = KMeans(n_clusters=k, random_state=24, init='random')
    clust.fit(df_scaled)
    
    #  silhouette score
    score = silhouette_score(df_scaled, clust.labels_)
    scores.append(score)


i_max = np.argmax(scores)

# Print the best number of clusters and its corresponding silhouette score
print("Best number of clusters:", Ks[i_max])
print("Best silhouette score:", scores[i_max])

# Fit KMeans clustering model with the best number of clusters
clust = KMeans(n_clusters=Ks[i_max], random_state=24)
clust.fit(df_scaled)

# Create a DataFrame with cluster labels
clust_data = pd.DataFrame(iris, columns=df.feature_names)
clust_data['Cluster'] = clust.labels_

# mean values of features for each cluster
print(clust_data.groupby('Cluster').mean())

# Access centroids
centroids = clust.cluster_centers_

# Create a DataFrame with cluster labels and centroids
centroids_df = pd.DataFrame(centroids, columns=df.feature_names)
centroids_df['Cluster'] = range(1, Ks[i_max]+1)

# Print centroids
print("Centroids:")
print(centroids_df)

