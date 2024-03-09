# -*- coding: utf-8 -*-
"""spaceship-titanic.ipynb

Original ipynb file is located at
    https://colab.research.google.com/drive/1nJAnzhxf3BlF4CTiAXnYlQXLQcQmYaU6
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import config
from data_management import load_dataset
from preprocessors import preprocess_boolean, label_encode, preprocess_cabin, preprocess_age
from model_evaluation import model_metrics
from models import model_rfc_train, model_rfc_predict, save_model

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

# Replace the null values in the categorical columns as MISSING
for col in config.FEATURES_TO_ENCODE:
  train[col] = train[col].fillna(train[col].mode()[0])
  test[col] = test[col].fillna(test[col].mode()[0])

# For numerical features, replace the null values with the median

for col in config.FEATURES_NUMERICAL:
  train[col] = train[col].fillna(train[col].median())
  test[col] = test[col].fillna(test[col].median())

# Label encode categorical columns
for col in config.FEATURES_TO_ENCODE:
  train = label_encode(train, col)
  test = label_encode(test, col)

# Drop the features which are not required
train = train.drop(config.FEATURES_DROP, axis=1)
test = test.drop(config.FEATURES_DROP, axis=1)

"""# Model Building"""
X = train.drop(config.TARGET, axis=1)
y = train[config.TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rfc = model_rfc_train(X_train, y_train)

y_pred_rfc = model_rfc_predict(X_test, rfc)

# Model evaluation
model_metrics(y_test, y_pred_rfc, 'Grid Searched Random Forest')

# A demo prediction to be used in gradio
test_demo = np.array([['Earth', 0, 'TRAPPIST-1e', 1,	100, 200, 100,	10, 10,	'C',	'2','S',	'10s']])
df_demo = pd.DataFrame(test_demo, columns=['HomePlanet',	'CryoSleep',	'Destination',	'VIP',	'RoomService'	,'FoodCourt'	,'ShoppingMall',	'Spa',	'VRDeck',	'deck',	'num'	,'side',	'AgeGroup'])

# label encode the input parameters
for col in config.FEATURES_TO_ENCODE:
  df_demo = label_encode(df_demo, col)

predict_demo = model_rfc_predict(df_demo, rfc)
print(f"The demo prediction is {predict_demo}")  

# Save the model
save_model(rfc)
