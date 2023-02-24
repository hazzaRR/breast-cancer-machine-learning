# STUDENT_ID: 100340391
# Created on: 03/11/2022
# Last update: [12/11/2022], [created information gain function and skeleton functions for gini and chi squared and rounded
# results to 2dp]
# Description: [a range of functions; information gain, gini index and chi squared that assess the quality of a split on
# a node]

import numpy as np

'''

for my 2x2 contingency table I will use a 2d array. The first inner array will contain the counts for the first
attribute value, with the first value in this inner array being the true class label and the second element being the
false class label. The second inner array will contain the counts for the second attribute value and are laid out the same

[ attribute1[True Count, False Count], attribute2[True Count, False Count]

for instance a 2x2 with these values 
           |True |False|
attribute 1|  7  |  1  |
attribute 2|  8  |  4  |

would appear like this:
[[7, 1],[8, 4]]

also for all calculations I have rounded my results to 2 decimal places

To handle total attribute values that equal zero, i have used ternary operators that check to see if a value is equal to
zero when trying to perform division, if this is the case the value is just set to zero instead of trying to divide by 
zero and crashing the program
In addition to this when it comes to log functions I have also used ternary operators again that sets the value to 1 in the 
log function if the the value equals zero since you cant log zero and you can handle this by using 1 as a value
'''


def get_information_gain(contingency_table):
    ''' Find the root node class label split '''
    true_counts = contingency_table[0][0] + contingency_table[1][0]
    false_counts = contingency_table[0][1] + contingency_table[1][1]

    '''Total number of data instances in the dataset'''
    total = true_counts + false_counts

    """
    calculation for Entropy for the root node 
    First find proportion of true and false classes at the root node and then use them in the entropy function
    """

    root_node_true_proportion = 0 if total == 0 or true_counts == 0 else np.round(true_counts / total, 2)
    root_node_false_proportion = 0 if total == 0 or false_counts == 0 else np.round(false_counts / total, 2)
    root_node_true_log_value = 1 if root_node_true_proportion == 0 else root_node_true_proportion
    root_node_false_log_value = 1 if root_node_false_proportion == 0 else root_node_false_proportion

    entropy = np.round(-((root_node_true_proportion * (np.log2(root_node_true_log_value))) \
                + root_node_false_proportion * np.log2(root_node_false_log_value)), 2)

    """calculation for Entropy for attribute 1 """
    a1_true_counts = contingency_table[0][0]
    a1_false_counts = contingency_table[0][1]
    a1_total = a1_true_counts + a1_false_counts

    a1_true_proportion = 0 if a1_total == 0 or a1_true_counts == 0 else np.round(a1_true_counts / a1_total, 2)
    a1_false_proportion = 0 if a1_total == 0 or a1_false_counts == 0 else np.round(a1_false_counts / a1_total, 2)
    a1_true_log_value = 1 if a1_true_proportion == 0 else a1_true_proportion
    a1_false_log_value = 1 if a1_false_proportion == 0 else a1_false_proportion

    a1 = np.round(-((a1_true_proportion * (np.log2(a1_true_log_value))) \
           + a1_false_proportion * np.log2(a1_false_log_value)), 2)


    """calculation for Entropy for attribute 2 """
    a2_true_counts = contingency_table[1][0]
    a2_false_counts = contingency_table[1][1]
    a2_total = a2_true_counts + a2_false_counts

    a2_true_proportion = 0 if a2_total == 0 or a2_true_counts == 0 else np.round(a2_true_counts / a2_total, 2)
    a2_false_proportion = 0 if a2_total == 0 or a2_false_counts == 0 else np.round(a2_false_counts / a2_total, 2)
    a2_true_log_value = 1 if a2_true_proportion == 0 else a2_true_proportion
    a2_false_log_value = 1 if a2_false_proportion == 0 else a2_false_proportion

    a2 = np.round(-((a2_true_proportion * (np.log2(a2_true_log_value))) \
           + a2_false_proportion * np.log2(a2_false_log_value)), 2)

    ''' get attribute proportions for the dataset'''
    if total == 0:
        a1_proportion = 0
        a2_proportion = 0

    else:
        a1_proportion = a1_total / total
        a2_proportion = a2_total / total

    ''' formula to work out the information gain of a given attribute'''
    information_gain = entropy - ((a1_proportion * a1) + (a2_proportion * a2))

    return np.round(information_gain, 2)


def get_gini(contingency_table):
    ''' Find the root node class label split '''
    true_counts = contingency_table[0][0] + contingency_table[1][0]
    false_counts = contingency_table[0][1] + contingency_table[1][1]

    '''Total number of data instances in the dataset'''
    total = true_counts + false_counts

    """
    calculation for Entropy for the root node 
    First find proportion of true and false classes at the root node and then use them in the entropy function
    """

    root_node_true_proportion = 0 if total == 0 else np.round(true_counts / total, 2)
    root_node_false_proportion = 0 if total == 0 else np.round(false_counts / total, 2)

    root_node_impurity = np.round(1 - (root_node_true_proportion ** 2 + root_node_false_proportion ** 2), 2)

    """calculation for gini impurity for attribute 1 """
    a1_true_counts = contingency_table[0][0]
    a1_false_counts = contingency_table[0][1]
    a1_total = a1_true_counts + a1_false_counts

    a1_true_proportion = 0 if a1_total == 0 else np.round(a1_true_counts / a1_total, 2)
    a1_false_proportion = 0 if a1_total == 0 else np.round(a1_false_counts / a1_total, 2)

    a1_impurity = np.round(1 - (a1_true_proportion ** 2 + a1_false_proportion ** 2), 2)

    """calculation for gini impurity for attribute 2 """
    a2_true_counts = contingency_table[1][0]
    a2_false_counts = contingency_table[1][1]
    a2_total = a2_true_counts + a2_false_counts

    a2_true_proportion = 0 if a2_total == 0 else np.round(a2_true_counts / a2_total, 2)
    a2_false_proportion = 0 if a2_total == 0 else np.round(a2_false_counts / a2_total, 2)

    a2_impurity = np.round(1 - (a2_true_proportion ** 2 + a2_false_proportion ** 2), 2)

    ''' get attribute proportions for the dataset'''
    a1_proportion = 0 if total == 0 else np.round(a1_total / total, 2)
    a2_proportion = 0 if total == 0 else np.round(a2_total / total, 2)

    ''' formula to work out the gini index of a given attribute'''
    gini_index = root_node_impurity - ((a1_proportion * a1_impurity) + (a2_proportion * a2_impurity))

    return np.round(gini_index, 2)


def get_chi_squared(contingency_table):
    ''' Find the root node class label split '''
    true_counts = contingency_table[0][0] + contingency_table[1][0]
    false_counts = contingency_table[0][1] + contingency_table[1][1]

    '''Total number of data instances in the dataset'''
    total = true_counts + false_counts

    """
    calculation for expected values for each value
    """
    root_node_true_proportion = 0 if total == 0 else true_counts / total
    root_node_false_proportion = 0 if total == 0 else false_counts / total

    a1_count = contingency_table[0][0] + contingency_table[0][1]
    a2_count = contingency_table[1][0] + contingency_table[1][1]

    a1_true = contingency_table[0][0]
    a1_false = contingency_table[0][1]
    a2_true = contingency_table[1][0]
    a2_false = contingency_table[1][1]

    a1_true_expected = 0 if root_node_true_proportion == 0 else np.round(a1_count * root_node_true_proportion, 2)
    a1_false_expected = 0 if root_node_false_proportion == 0 else np.round(a1_count * root_node_false_proportion, 2)
    a2_true_expected = 0 if root_node_true_proportion == 0 else np.round(a2_count * root_node_true_proportion, 2)
    a2_false_expected = 0 if root_node_false_proportion == 0 else np.round(a2_count * root_node_false_proportion, 2)

    """Find the (O-E)2 / E for each observed value in the contingency table"""

    val1 = 0 if a1_true_expected == 0 else ((a1_true - a1_true_expected) ** 2) / a1_true_expected
    val2 = 0 if a1_false_expected == 0 else ((a1_false - a1_false_expected) ** 2) / a1_false_expected
    val3 = 0 if a2_true_expected == 0 else ((a2_true - a2_true_expected) ** 2) / a2_true_expected
    val4 = 0 if a2_false_expected == 0 else ((a2_false - a2_false_expected) ** 2) / a2_false_expected

    x2 = val1 + val2 + val3 + val4

    return np.round(x2, 2)


headache_contingency_table = [[4, 2], [3, 5]]

print(f"measure_information_gain = {get_information_gain(headache_contingency_table)}")
print(f"measure_gini_index = {get_gini(headache_contingency_table)}")
print(f"measure_chi_squared = {get_chi_squared(headache_contingency_table)}")


with open('../output/headache_splitting_diagnosis.txt ', 'w') as f:
    f.write(f"measure_information_gain = {get_information_gain(headache_contingency_table)}\n")
    f.write(f"measure_gini_index = {get_gini(headache_contingency_table)}\n")
    f.write(f"measure_chi_squared = {get_chi_squared(headache_contingency_table)}\n")

