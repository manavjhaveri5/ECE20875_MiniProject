from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from MiniProjectPath1 import dataset_1_filtered
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Select relevant features
features = dataset_1_filtered[['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']]

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Run KMeans with 6 clusters
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(features_scaled)

# Assign the cluster labels back to the original DataFrame
dataset_1_filtered['Cluster'] = kmeans.labels_

# Run PCA and reduce the data to two components for visualization
pca = PCA(n_components=2)
features_reduced = pca.fit_transform(features_scaled)

# Plot the reduced data with cluster labels
plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=dataset_1_filtered['Cluster'], cmap='viridis', alpha=0.5)
plt.title('Clusters of Student Video-Watching Behaviors')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()
