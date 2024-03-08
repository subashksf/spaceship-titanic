# -*- coding: utf-8 -*-
"""spaceship-titanic.ipynb

Original ipynb file is located at
    https://colab.research.google.com/drive/1nJAnzhxf3BlF4CTiAXnYlQXLQcQmYaU6
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import config
from data_management import load_dataset
from preprocessors import preprocess_boolean, label_encode, preprocess_cabin, preprocess_age

train = load_dataset(config.TRAIN_FILE)
test = load_dataset(config.TEST_FILE)

"""# Feature Engineering"""

# Cabin consists of deck/num/side. Let us derive those values, and drop the cabin feature
train = preprocess_cabin(train)
test = preprocess_cabin(test)

# Bucket the values in the age column
train = preprocess_age(train)
test = preprocess_age(test)

# Replace bool values in features with 0-false, 1-true
for col in config.FEATURES_BOOL:
  preprocess_boolean(train, col)
  preprocess_boolean(test, col)

# Replace bool values in the label with 0-false, 1-true
preprocess_boolean(train, config.TARGET)

# Replace the null values in the categorical columns with the mode

for col in config.FEATURES_TO_ENCODE:
  train[col].fillna(train[col].mode()[0], inplace=True)
  test[col].fillna(test[col].mode()[0], inplace=True)

# For numerical features, replace the null values with the median

for col in config.FEATURES_NUMERICAL:
  train[col].fillna(train[col].median(), inplace=True)
  test[col].fillna(test[col].median(), inplace=True)

# Label encode categorical columns
for col in config.FEATURES_TO_ENCODE:
  train = label_encode(train, col)
  test = label_encode(test, col)

# Drop the features which are not required
train.drop(config.FEATURES_DROP, axis=1, inplace=True)
test.drop(config.FEATURES_DROP, axis=1, inplace=True)

"""# Model Building"""

# Model evaluation
from sklearn import metrics

def model_metrics(prediction, model):

  # Model Accuracy, how often is the classifier correct?
  print(f"{model} Accuracy:",metrics.accuracy_score(y_test, prediction))
  print("\n")

  # confusion matrix
  cm= metrics.confusion_matrix(y_test, prediction)
  cm_dis = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
  cm_dis.plot()
  #plt.show()

  classx= metrics.classification_report(y_test, prediction)
  print("\n")
  print(f"{model} Classification Report:\n", classx)

X = train.drop(config.TARGET, axis=1)
y = train[config.TARGET]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model using the best random forest parameters. Refer the ipynb file for details of the GridSearchCV
rfc1=RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators= 500, max_depth=8, criterion='entropy')
rfc1.fit(X_train, y_train)

y_pred_rf_best=rfc1.predict(X_test)

# Model evaluation
model_metrics(y_pred_rf_best, 'Grid Searched Random Forest')
