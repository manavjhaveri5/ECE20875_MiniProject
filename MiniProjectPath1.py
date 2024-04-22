import matplotlib.pyplot as plt
import pandas
import numpy as np
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
from sklearn.metrics import accuracy_score, classification_report




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

##################################################LinearRegression############################################

# X = dataset_1_filtered[['fracSpent','fracComp','fracPlayed','fracPaused','numPauses','avgPBR','stdPBR','numRWs','numFFs']]
# y = dataset_1_filtered[['s']]
# X_train, X_test, y_train, y_test = train_test_split(X,y,random_state =0)

# mean_X = np.mean(X_train, axis=0)
# std_X = np.std(X_train, axis=0)
# mean_y = np.mean(y)
# std_y = np.std(y)
# X_train_norm = (X_train - mean_X) / std_X
# X_test_norm = (X_test - mean_X) / std_X
# y_train_norm = (y_train - mean_y) / std_y


# alpha = 0.5  # Regularization strength
# ridge_model = Ridge(alpha=alpha)
# ridge_model.fit(X_train_norm, y_train_norm)


# # # Predict on test set
# y_pred_test = ridge_model.predict(X_test_norm)


# # Evaluate model
# mse = mean_squared_error(y_test, y_pred_test)
# r2 = r2_score(y_test, y_pred_test)
# print('Mean squared error with normalization:', mse)
# print('Coefficient of Determination:', r2)


################################################Prediction###############################################
# features = ['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']
# X = dataset_1[features]
# Y = dataset_1[['s']]
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# # Initialize and train 
# knn = KNeighborsClassifier(n_neighbors=5)
# svm = SVC(kernel='linear', probability=True)  # Linear kernel
# mlp = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
# knn.fit(X_train, y_train)
# svm.fit(X_train, y_train)
# mlp.fit(X_train, y_train)

# # Predict on the test set
# knn_pred = knn.predict(X_test)
# svm_pred = svm.predict(X_test)
# mlp_pred = mlp.predict(X_test)

# # Evaluate models
# print("KNN Classification Report:")
# print(classification_report(y_test, knn_pred))
# print("SVM Classification Report:")
# print(classification_report(y_test, svm_pred))
# print("MLP Classification Report:")
# print(classification_report(y_test, mlp_pred))

# # Calculate ROC AUC Scores
# print("KNN ROC AUC Score:", roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1]))
# print("SVM ROC AUC Score:", roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1]))
# print("MLP ROC AUC Score:", roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1]))

def perform_logistic_regression(X_train, X_test, y_train, y_test):
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        print(f"Skipping logistic regression for Video ID {video_id} due to only one unique class.")
        return None, None

    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)

    y_pred = logistic.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: ", acc)

    print("Classification report: ", classification_report(y_test, y_pred))

    return y_pred, acc

msevals = []
vdids = []
acc_list = []

for video_id in dataset_1_filtered['VidID'].unique():
    
    video_data = dataset_1_filtered[dataset_1_filtered['VidID'] == video_id]

    if len(video_data) < 2:
        print(f"Not enough samples for Video ID {video_id}")
        continue


    X_video = video_data[['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']]
    y_video = video_data['s']

    X_train_video, X_test_video, y_train_video, y_test_video = train_test_split(X_video, y_video, random_state=0)

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_video, y_train_video)

    y_pred_video = ridge_model.predict(X_test_video)

    # logistic = LogisticRegression()
    # logistic.fit(X_train_video, y_train_video)

    # y_pred = logistic.predict(X_test_video)

    # acc = accuracy_score(y_test_video, y_pred)
    y_pred,acc = perform_logistic_regression(X_train_video, X_test_video, y_train_video, y_test_video)

    print("Accuracy: ", acc)

    #print("Classification report: ", classification_report(y_test_video, y_pred))

    mse_video = mean_squared_error(y_test_video, y_pred_video)

    if mse_video<1000:
        msevals.append(mse_video)
        vdids.append(video_id)
        acc_list.append(acc)


    print('Mean Squared Error for Video ID', video_id, ':', mse_video)
    print()

plt.scatter(vdids, msevals)
plt.xlabel('Video IDs')
plt.ylabel('MSE Value')
plt.title('MSE Values for each Video')
plt.show()

# Plot accuracy vs. MSE
plt.scatter(acc_list, msevals)
plt.xlabel('Accuracy')
plt.ylabel('MSE Value')
plt.title('MSE Values vs. Accuracy')
plt.show()


accl = [acc for acc in acc_list if acc is not None]
Mseavg = sum(msevals) / len(msevals)
accavg = sum(accl)/ len(accl)
print("Average MSE Value: ", Mseavg)
print("Average Accuracy Value: ", accavg)
1
############################################################################################LOGISTIC REGRESSION#############################################################################################

# X= video_data[['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']]
# y= video_data['s']

# X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)

# logistic = LogisticRegression()
# logistic.fit(X_train, y_train)

# y_pred = logistic.predict(X_test)

# acc = accuracy_score(y_test, y_pred)
# print("Accuracy: ", acc)

# print("Classification report: ", classification_report(y_test,y_pred))