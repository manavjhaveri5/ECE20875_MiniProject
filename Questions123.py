from Classifiers import Classify
from LinearLogRegression import Linear_Log_Regression
from MiniProjectPath1 import dataset_1_filtered

#THE FOLLOWING CODE CAN BE RUN TO PROVIDE INSIGHT FOR EACH QUESTION

#QUESTION 2
Classify(dataset_1_filtered)

#QUESTION 2 AND 3 (Provides Linear Rgression and MSE value for how predictable each students' score is for each video, also does Logistic Regression for the 's' class and provides accuracy score for each video)
Linear_Log_Regression(dataset_1_filtered)