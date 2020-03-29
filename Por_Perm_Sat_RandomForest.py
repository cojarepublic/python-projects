# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:13:27 2020

@author: CHUKWUJIOKE
"""


# Importing the libraries
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Importing the dataset
dataset = pd.read_csv('Por_Perm_Sat.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)

# Making the Random Forest Regression model
model = RandomForestRegressor()
model.fit(X_train,y_train)

# Get the mean absolute error on the validation data
predicted_saturations = model.predict(X_test)
MAE = mean_absolute_error(y_test , predicted_saturations)
print('Random forest validation MAE = ', MAE)

# Making predictions
predicted_saturations = model.predict(X_test)