import matplotlib.pyplot as plt
import pandas
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

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
kmeans = KMeans(n_clusters=5, random_state = 0)
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

##################################################LinearRegression############################################3

X = dataset_1_filtered[['fracSpent','fracComp','fracPlayed','fracPaused','numPauses','avgPBR','stdPBR','numRWs','numFFs']]
y = dataset_1_filtered[['s']]

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state =0)

mean_X = np.mean(X_train, axis=0)
std_X = np.std(X_train, axis=0)
mean_y = np.mean(y)
std_y = np.std(y)
X_train_norm = (X_train - mean_X) / std_X
X_test_norm = (X_test - mean_X) / std_X
y_train_norm = (y_train - mean_y) / std_y

# Train Ridge regression model
alpha = 0.50  # Regularization strength
ridge_model = Ridge(alpha=alpha)
ridge_model.fit(X_train_norm, y_train)

# Predict on test set
y_pred_test = ridge_model.predict(X_test_norm)

# Evaluate model
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print('Mean squared error with normalization:', mse)
print('Coefficient of Determination:', r2)