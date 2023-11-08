#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:48:06 2023

@author: aarabhidatta
"""

import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
spotify_data=pd.read_excel('SpotifyDataSet.xlsx', na_values=[0])
spotify_data_clean=spotify_data.dropna(subset=['valence','loudness','danceability','speechiness'])
spotify_data_clean.reset_index(drop=True, inplace=True)
features=['region','language','valence','loudness','danceability','speechiness','artist_genre']
target='streams'
missing_values = spotify_data_clean[['region','language','valence','loudness','danceability','speechiness','artist_genre', 'streams']].isnull().sum()
print(missing_values)
X=spotify_data_clean[features]
y=spotify_data_clean[target]
X=pd.get_dummies(X, columns=['region','language','artist_genre'])
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
#initializing the linear regression model 
model=LinearRegression()
# training the model on the training data
model.fit(X_train, y_train)
#make predictions on the test data
predictions=model.predict(X_test)
#evaluate the model (for example, using mean squared error)
mse=mean_squared_error(y_test,predictions)
print('Mean Squared Error: ', mse)



