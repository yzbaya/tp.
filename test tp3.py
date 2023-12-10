# -*- coding: utf-8 -*-
"""Bienvenue dans Colaboratory

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/notebooks/intro.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

dataset = pd.read_csv('car_evaluation.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
dataset.head()

dataset.shape

dataset.info()

# Rename the column
col_names = ['buying', 'meant', 'doors', 'persons', 'lug_boot', 'safety', 'class']

dataset.columns = col_names
col_names

for col in col_names:
    print(dataset[col].value_counts())

dataset['class'].value_counts()

#method is used to check for missing or NaN (Not-a-Number) values in the DataFrame.
dataset.isnull().sum()

X = dataset.drop(['class'], axis=1)
y = dataset['class']

X.head()

y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=42)
X_train.shape, X_test.shape

y_train.shape, y_test.shape

X_train.dtypes

X_train.head()

pip install category_encoders

import category_encoders as ce

# encode variables with ordinal encoding
encoder = ce.OrdinalEncoder(cols=['buying', 'meant', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

X_train

X_test

X_train.head()

X_test.head()

#Decision Tree Criterion with gini index

from sklearn.tree import DecisionTreeClassifier

# typical example of using a decision tree classifier in a machine learning task.
#'DecisionTreeClassifier': builds a decision tree model to predict the class labels of data points.
#The 'gini' criterion is one of the options and is used to build decision trees based on the Gini impurity.
#clf_gini.fit(X_train, y_train):This line of code trains the decision tree classifier (clf_gini) on your training data.
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, y_train)

y_pred_gini = clf_gini.predict(X_test)
y_pred_gini[:5]

# Check Accuracy score
from sklearn.metrics import accuracy_score

print("Model Accuracy score with criterion gini index {0:0.4f}".format(accuracy_score(y_pred_gini, y_test)))

y_pred_train_gini = clf_gini.predict(X_train)
y_pred_train_gini

print("Model Accuracy score with criterion gini index {0:0.4f}".format(accuracy_score(y_pred_train_gini, y_train)))

#The code you provided is used to calculate and print the accuracy scores of a machine learning model using the Gini index criterion for both
#the test dataset and the training dataset. The accuracy score is a common metric for evaluating the performance of classification models.
print("Model Accuracy score with criterion gini index for test dataset {0:0.4f}".format(accuracy_score(y_pred_gini, y_test)))
print("Model Accuracy score with criterion gini index for train dataset {0:0.4f}".format(accuracy_score(y_pred_train_gini, y_train)))

#The code you provided is used to visualize the decision tree that was trained with the Gini impurity criterion on the training dataset.
#It utilizes the tree.plot_tree function from Scikit-Learn to generate a graphical representation of the decision tree
plt.figure(figsize=(10, 8))
from sklearn import tree
tree.plot_tree(clf_gini.fit(X_train, y_train))

#The code you provided is used to visualize a decision tree using the Graphviz library.
import graphviz
dot_data = tree.export_graphviz(clf_gini, out_file=None, feature_names=X_train.columns,  class_names=y_train,  filled=True, rounded=True,  special_characters=True)
graph = graphviz.Source(dot_data)
graph