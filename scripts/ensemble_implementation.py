# STUDENT_ID: 100340391
# Created on: 16/11/2022
# Last update:
# [24/11/2022], [built the five classifiers and used the Voting classifier to create a ensemble. Then got the majority vote
# using the predict method]
# Description: [implementing a ensemble classifier of 5 decision tree classifiers using the sklearn library and
# a range of different splitting measures and max depths]

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_test_data(data):
    """ Function that returns a 70:30 training and testing sets of data, which first removes all the
        null, undefined, unknown and ? values from the data set and turns all categorical values into numeric ones
    """

    """ drops any data in the dataframe where values are null, undefined or NaN """
    data = data.dropna()
    """ goes through all the data in each column and drops any rows where the value is equal to ? """
    for col in data.columns:
        data.drop(data[data[col] == "?"].index, inplace=True)

    # reset dataframe indexes
    data = data.reset_index(drop=True)

    """ replace categorical values with numeric values for every column"""
    for col in data.columns:

        # if column is an object type replace the categorical values with numeric values
        if data[col].dtype == np.object_:
            # find the unique labels of each column
            col_values = data[col].unique()

            # make an array of values from 0 to n of the length of unique labels in a column
            numeric_values = np.arange(0, len(data[col].unique()))

            # replace categorical values with numerical representation
            data.replace(col_values, numeric_values, inplace=True)

    """ slice data into attributes and class values"""
    X = data.iloc[:, 1:12].values
    y = data[0].values

    """randomly shuffles the data and split it into a 75:25 split of train:test data """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    return X_train, X_test, y_train, y_test


breast_cancer_data = pd.read_csv('../data/breast-cancer.data', header=None)
X_train, X_test, y_train, y_test = train_test_data(breast_cancer_data)

""" decision trees of ranging max depths and criterion """
decision_tree_1 = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=1)
decision_tree_2 = DecisionTreeClassifier(criterion="entropy", max_depth=15, random_state=1)
decision_tree_3 = DecisionTreeClassifier(criterion="gini", max_depth=20, random_state=1)
decision_tree_4 = DecisionTreeClassifier(criterion="entropy", max_depth=25, random_state=1)
decision_tree_5 = DecisionTreeClassifier(criterion="gini", max_depth=30, random_state=1)


""" ensemble the 5 classifiers together using the Voting Classifier method"""
dt_ensemble = VotingClassifier(estimators=[('dt1', decision_tree_1), ('dt2', decision_tree_2), ('dt3', decision_tree_3), \
                                           ('dt4', decision_tree_4), ('dt5', decision_tree_5)])

""" train the ensemble with the train data"""
dt_ensemble = dt_ensemble.fit(X_train, y_train)

""" 
get the predictions of the ensemble using the predict method, which uses majority vote to get a single prediction
for each data instance
"""
ensemble_predictions = dt_ensemble.predict(X_test)


""" get the balance accuracy using the balance accuracy score"""
ensemble_bal_acc = balanced_accuracy_score(y_test, ensemble_predictions)

print(f"ensemble_balanced_acc = {ensemble_bal_acc}")

with open('../output/ensemble_balanced_accuracy.txt', 'w') as f:
    f.write(f"ensemble_balanced_acc = {ensemble_bal_acc}\n")
