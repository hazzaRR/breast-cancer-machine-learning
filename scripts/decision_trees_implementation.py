# STUDENT_ID: 100340391
# Created on: 10/11/2022
# Last update:
# [21/11/2022], [created decision tree with different max depths and computed the balance accuracy for each one.
# Then created tree based off the decision tree with the max_depth with the highest balance accuracy]
# Description: [implementing a decision tree classifier using the sklearn library]

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn import tree


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

    """randomly shuffles the data and split it into a 70:30 split of  train:test data """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    return X_train, X_test, y_train, y_test


breast_cancer_data = pd.read_csv('../data/breast-cancer.data', header=None)
X_train, X_test, y_train, y_test = train_test_data(breast_cancer_data)


""" decision tree that has a max depth of 3 """
decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=1)
decision_tree.fit(X_train, y_train)

test_predict = decision_tree.predict(X_test)
train_predict = decision_tree.predict(X_train)

print(f"dt_balanced_acc_test = {balanced_accuracy_score(y_test, test_predict)}")
print(f"dt_balanced_acc_train = {balanced_accuracy_score(y_train, train_predict)}")


with open('../output/dt_balanced_acc.txt', 'w') as f:
    f.write(f"dt_balanced_acc_test = {balanced_accuracy_score(y_test, test_predict)}\n")
    f.write(f"dt_balanced_acc_train = {balanced_accuracy_score(y_train, train_predict)}\n")

bal_accuracies = []

max_tree_depth = np.arange(1, 11)

best_bal_accuracy_score = 0
best_tree_depth = 0
best_tree = 0

with open('../output/dt_balanced_acc_scores.txt', 'w') as f:
    for tree_depth in max_tree_depth:
        decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=tree_depth, random_state=1)
        decision_tree.fit(X_train, y_train)

        test_predict = decision_tree.predict(X_test)

        bal_acc = balanced_accuracy_score(y_test, test_predict)

        if bal_acc > best_bal_accuracy_score:
            best_bal_accuracy_score = bal_acc
            best_tree = decision_tree
            best_tree_depth = tree_depth

        bal_accuracies.append(bal_acc)

        print(f"dt_entropy_max_depth_{tree_depth}_balanced_acc = {bal_acc}")
        f.write(f"dt_entropy_max_depth_{tree_depth}_balanced_acc = {bal_acc}\n")


tree.plot_tree(best_tree)
plt.savefig(f'../output/dt_entropy_max_depth_{best_tree_depth}.png')
plt.show()

plt.plot(max_tree_depth, bal_accuracies)
plt.ylabel('Balance Accuracy')  # xlabel
plt.xlabel('Max Tree Depth')  # ylabel

plt.savefig(f'../output/dt_balanced_acc_scores.png')

plt.show()
