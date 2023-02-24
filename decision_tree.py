from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn import linear_model


drop_col = [ "Topic", "Class", "Stratification2", "Stratification1", "Topic", "LocationDesc", "Question"]
df = pd.read_csv('data.csv', usecols=lambda x: x in drop_col) 
# df = df.sample(frac=.3)
df = df.replace(',','', regex=True)
df = df.dropna()

class_labels, class_uniques = pd.factorize(df["Class"])
strat_one_labels, strat_one_uniques = pd.factorize(df["Stratification1"])
strat_two_labels, strat_two_uniques = pd.factorize(df["Stratification2"])
loc_labels, loc_uniques = pd.factorize(df["LocationDesc"])
transactions = []
for x in range(len(class_labels)):
    templist = [class_labels[x]]
    templist.append(strat_one_labels[x])
    templist.append(strat_two_labels[x])
    templist.append(loc_labels[x])
    transactions.append(templist)
    # transactions.append([class_labels[x]+","+strat_one_labels[x]+","+strat_two_labels[x]])

print(len(transactions))
print(len(df["Topic"]))

# # target is what we want to predict
# # fit( data, target)
# # target is the value predicted
# # data is the data in line with the target
# # 1 target per row of data.

# #we'll use Stratification1 (AGE) Stratification2 (RACE/GENDER) , CLASS to predict TOPIC
clf = tree.DecisionTreeClassifier(max_leaf_nodes=12, max_depth = 6)
clf = clf.fit(transactions, df["Topic"])
tree.plot_tree(clf)
# plt.show()


import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=["Class", "AGE", "RACE/GENDER", "LocationDesc"], class_names=list(df["Topic"].unique()),  rounded=True,  
                     special_characters=True, impurity=True, filled=True) 
graph = graphviz.Source(dot_data) 
graph.render("Alzheimers_Indicators_Decision_Tree_5", format='png') 