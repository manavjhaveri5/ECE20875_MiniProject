import matplotlib.pyplot as plt
import pandas
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score




'''
 The following is the starting code for path1 for data reading to make your first step easier.
 'dataset_1' is the clean data for path1.
'''

with open('behavior-performance.txt','r') as f:
    raw_data = [x.strip().split('\t') for x in f.readlines()]
df = pandas.DataFrame.from_records(raw_data[1:],columns=raw_data[0])
df['VidID']       = pandas.to_numeric(df['VidID'])
df['fracSpent']   = pandas.to_numeric(df['fracSpent'])
df['fracComp']    = pandas.to_numeric(df['fracComp'])
df['fracPlayed']  = pandas.to_numeric(df['fracPlayed'])
df['fracPaused']  = pandas.to_numeric(df['fracPaused'])
df['numPauses']   = pandas.to_numeric(df['numPauses'])
df['avgPBR']      = pandas.to_numeric(df['avgPBR'])
df['stdPBR']      = pandas.to_numeric(df['stdPBR'])
df['numRWs']      = pandas.to_numeric(df['numRWs'])
df['numFFs']      = pandas.to_numeric(df['numFFs'])
df['s']           = pandas.to_numeric(df['s'])
dataset_1 = df
#print(dataset_1[0:1].to_string())  #This line will print out the first 35 rows of your data

# QUESTION 1 

video_completion = df[df['fracComp'] >= 0.9] #Filtering for fracComp >= 0.9

video_counts = video_completion.groupby('userID')['VidID'].nunique() #Counts how many videos each unique user has completed

users_with_5_videos = video_counts[video_counts >= 5].index #Filters to those who have completed 5 videos

dataset_1_filtered = video_completion[video_completion['userID'].isin(users_with_5_videos)] #Final Filtered Dataframe

dataset_1_filteredArray = dataset_1_filtered.to_numpy() #Final Filtered array

print('Shape of Original Dataset: ' ,dataset_1.shape)
print('Shape of Filtered Dataset: ' ,dataset_1_filteredArray.shape)

Desired_Features = dataset_1_filtered[['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']]
X = np.array(Desired_Features)
print('Shape of Filtered Dataset with Desired Features: ', X.shape )

####################################################################KMEANS##############################################
kmeans = KMeans(n_clusters=3, random_state = 0)
kmeans.fit(X)
print(kmeans.cluster_centers_)
print(kmeans.labels_)

dataset_1_filtered['Cluster'] = kmeans.labels_

# Run PCA and reduce the data to two components for visualization
pca = PCA(n_components=2)
features_reduced = pca.fit_transform(X)


plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=dataset_1_filtered['Cluster'])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Visualize Clusters for Desired Two Features
dataset_1_filtered['Cluster'] = kmeans.labels_

x = 'fracSpent'
y = 'fracPaused'

plt.scatter(Desired_Features[x], Desired_Features[y], c=kmeans.labels_)
plt.title('Clusters of Student Video-Watching Behaviors')
plt.xlabel(x)
plt.ylabel(y)
plt.show()

features = ['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']

# Analyzing cluster characteristics
cluster_characteristics = dataset_1_filtered.groupby('Cluster')[features].mean()
print("\nCluster Characteristics:")
print(cluster_characteristics)


##################################################LinearRegression############################################3

X = dataset_1[['fracSpent','fracComp','fracPlayed','fracPaused','numPauses','avgPBR','stdPBR','numRWs','numFFs']]
y = dataset_1[['s']]

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state =0)

mean_X = np.mean(X_train, axis=0)
std_X = np.std(X_train, axis=0)
mean_y = np.mean(y)
std_y = np.std(y)
X_train_norm = (X_train - mean_X) / std_X
X_test_norm = (X_test - mean_X) / std_X
y_train_norm = (y_train - mean_y) / std_y

# Train Ridge regression model
alpha = 0.5  # Regularization strength
ridge_model = Ridge(alpha=alpha)
ridge_model.fit(X_train_norm, y_train)

# Predict on test set
y_pred_test = ridge_model.predict(X_test_norm)

# Evaluate model
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print('Mean squared error with normalization:', mse)
print('Coefficient of Determination:', r2)


################################################Prediction###############################################
features = ['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']
X = dataset_1[features]
Y = dataset_1[['s']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize and train 
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear', probability=True)  # Linear kernel
mlp = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
mlp.fit(X_train, y_train)

# Predict on the test set
knn_pred = knn.predict(X_test)
svm_pred = svm.predict(X_test)
mlp_pred = mlp.predict(X_test)

# Evaluate models
print("KNN Classification Report:")
print(classification_report(y_test, knn_pred))
print("SVM Classification Report:")
print(classification_report(y_test, svm_pred))
print("MLP Classification Report:")
print(classification_report(y_test, mlp_pred))

# Calculate ROC AUC Scores
print("KNN ROC AUC Score:", roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1]))
print("SVM ROC AUC Score:", roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1]))
print("MLP ROC AUC Score:", roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1]))