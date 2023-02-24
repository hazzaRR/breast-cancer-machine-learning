# STUDENT_ID: 100340391
# Created on: 16/11/2022
# Last update:
# [24/11/2022], [trained optimised models on the train_validation set and got the accuracy score after fitting the models
# on the test set]
# Description: [file that implements a range of classifiers and while tuning their parameters to find the best model
# for the breast cancer dataset with real values]


import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

""" load in breast cancer real value data set and split into into train and testing sets """
X, y = load_breast_cancer(return_X_y=True)
X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

k_fold = KFold(n_splits=5, shuffle=False)
""" parameter values for decision trees """
dt_max_depth_param = [2, 3, 5, 7, 10]


""" variables to store the best decision tree depth, all the median accuracies and then the highest median accuracy """
best_dt_depth = 0
median_df_accuracies = []
highest_median_dt_accuracy = 0

""" cross validation on decision tree classifiers """
for depth in dt_max_depth_param:

    dt = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=1)

    """ do a 5-fold cross validation on the decision tree """
    scores = cross_val_score(dt, X_train_validation, y_train_validation,
                             cv=k_fold, scoring='accuracy', verbose=False)
    """ get the median accuracy from the 5 fold cross validation scores """
    median_acc = np.median(scores)
    median_df_accuracies.append(median_acc)

    """ 
    check to see if the median accuracy from the last decision tree params is better then previous best
    also store the params values that achieve the highest median accuracy 
    """

    if median_acc > highest_median_dt_accuracy:
        highest_median_dt_accuracy = median_acc
        best_dt_depth = depth

print(f"decision tree highest median value = {highest_median_dt_accuracy} with the best depth being = {best_dt_depth}")
print(f"decision tree median accuracy scores: {median_df_accuracies}")

""" cross validation on random forest classifiers """

""" parameter values for random forest """
rf_estimator_params = [100, 200, 500]


"""
variables to store the best random forest estimator, all the median accuracies and then the highest median 
accuracy 
"""
best_rf_estimator = 0
median_rf_accuracies = []
highest_median_rf_accuracy = 0

for estimator in rf_estimator_params:

    rf = RandomForestClassifier(n_estimators=estimator, random_state=1)

    """ do a 5-fold cross validation on the random forest classifier """
    scores = cross_val_score(rf, X_train_validation, y_train_validation,
                             cv=k_fold, scoring='accuracy', verbose=False)

    """ get the median accuracy from the 5 fold cross validation scores """
    median_acc = np.median(scores)
    median_rf_accuracies.append(median_acc)

    """
    check to see if the median accuracy from the last random forest params is better then previous best
    also store the params values that achieve the highest median accuracy
    """
    if median_acc > highest_median_rf_accuracy:
        highest_median_rf_accuracy = median_acc
        best_rf_estimator = estimator

print(f"random forest highest median value = {highest_median_rf_accuracy} and best estimators = {best_rf_estimator}")
print(f"random forest median accuracy scores: {median_rf_accuracies}")

""" cross validation on K-NN classifiers """

""" parameter values for K-NN classifier """
knn_k_values = [1, 11, 21, 31, 51]
knn_metrics = ["euclidean", "manhattan"]


""" 
variables to store the best K value, best distance metric and all the median accuracies 
and then the highest median accuracy 
"""
best_knn_k_value = 0
best_knn_metric = ""
median_knn_accuracies = []
highest_median_knn_accuracy = 0


for metric_value in knn_metrics:

    median_accuracies = []
    for k_value in knn_k_values:
        knn = KNeighborsClassifier(n_neighbors=k_value, metric=metric_value)

        """ do a 5-fold cross validation on the nearest_neighbour classifier """
        scores = cross_val_score(knn, X_train_validation, y_train_validation,
                                 cv=k_fold, scoring='accuracy', verbose=False)
        """ get the median accuracy from the 5 fold cross validation scores """
        median_acc = np.median(scores)
        median_accuracies.append(median_acc)

        """ 
        check to see if the median accuracy from the last  params is better then any others
        also store the params values that achieve the highest median accuracy
        """
        if median_acc > highest_median_knn_accuracy:
            highest_median_knn_accuracy = median_acc
            best_knn_k_value = k_value
            best_knn_metric = metric_value

    median_knn_accuracies.append(median_accuracies)

print(f"K-NN highest median value = {highest_median_knn_accuracy}, with the best k value being = {best_knn_k_value} and the best metric = {best_knn_metric}")

print(f"K-NN median accuracy scores: {median_knn_accuracies}")

""" build optimised classifiers on the best parameters and test on test set"""

optimised_dt = DecisionTreeClassifier(criterion="entropy", max_depth=best_dt_depth, random_state=1)
optimised_dt.fit(X_train_validation, y_train_validation)
optimised_dt_predictions = optimised_dt.predict(X_test)
optimised_dt_acc = accuracy_score(y_test, optimised_dt_predictions)


optimised_rf = RandomForestClassifier(n_estimators=best_rf_estimator, random_state=1)
optimised_rf.fit(X_train_validation, y_train_validation)
optimised_rf_predictions = optimised_rf.predict(X_test)
optimised_rf_acc = accuracy_score(y_test, optimised_rf_predictions)

optimised_knn = KNeighborsClassifier(n_neighbors=best_knn_k_value, metric=best_knn_metric)
optimised_knn.fit(X_train_validation, y_train_validation)
optimised_knn_predictions = optimised_knn.predict(X_test)
optimised_knn_acc = accuracy_score(y_test, optimised_knn_predictions)

print(f"Optimised Decision tree accuracy:{optimised_dt_acc}")
print(f"Optimised Random forest accuracy:{optimised_rf_acc}")
print(f"Optimised K nearest-neighbours accuracy:{optimised_knn_acc}")


print(max(optimised_dt_acc, optimised_rf_acc, optimised_knn_acc))

