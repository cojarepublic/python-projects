# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:00:15 2020

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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Making the DNN model
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(17, kernel_initializer='normal',input_dim = 2 , activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(35, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(35, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(35, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

# Define a checkpoint
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

# Training the model
NN_model.fit(X_train, y_train, epochs = 6, validation_split = 0.2, callbacks=callbacks_list)

# Making the predictions
predictions = NN_model.predict(X_test)
