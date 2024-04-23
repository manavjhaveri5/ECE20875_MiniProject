from MiniProjectPath1 import dataset_1_filtered
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report

def Linear_Log_Regression(dataset_1_filtered):
    def perform_logistic_regression(X_train, X_test, y_train, y_test):
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            #print(f"Skipping logistic regression for Video ID {video_id} due to only one unique class.")
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
        
        y_pred,acc = perform_logistic_regression(X_train_video, X_test_video, y_train_video, y_test_video)

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
