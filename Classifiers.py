from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import classification_report


def Classify(dataset_1):
    features = ['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']
    X = dataset_1[features]
    Y = dataset_1[['s']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

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