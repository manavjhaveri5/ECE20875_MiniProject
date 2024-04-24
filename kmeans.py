from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from MiniProjectPath1 import dataset_1_filtered
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples , silhouette_score
import pandas as pd
import seaborn as sns




# Select relevant features
features = dataset_1_filtered[['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']]

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Run KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(features_scaled)

# Assign the cluster labels back to the original DataFrame
dataset_1_filtered = dataset_1_filtered.copy()
dataset_1_filtered.loc[:, 'Cluster'] = kmeans.labels_

# Run PCA and reduce the data to three components for visualization
pca = PCA(n_components=3)
features_reduced = pca.fit_transform(features_scaled)

# Plot the reduced data with cluster labels
plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=dataset_1_filtered['Cluster'], cmap='viridis', alpha=0.5)
plt.title('Clusters of Student Video-Watching Behaviors')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()

X, y = make_blobs(n_samples=len(dataset_1_filtered),cluster_std=1,centers=4,n_features=2, shuffle=True,center_box=(-10.0,10.0),random_state=2)
plt.scatter(X[:, 0], X[:, 1], alpha=0.8)
plt.title("Optimal Number of Clusters")
plt.show()

clusterer = KMeans(n_clusters=3, n_init=10, random_state=42)
cluster_labels = clusterer.fit_predict(features_scaled)
silhouette_avg = silhouette_score(features_scaled, cluster_labels)

print(f"The average silhouette score for n_clusters = 3 is ",silhouette_avg)




