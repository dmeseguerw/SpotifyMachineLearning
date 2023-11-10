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
import matplotlib.pyplot as plt
spotify_data=pd.read_csv('SpotifyDataSet.csv', na_values=[0])

cols_to_encode = ['week','artist_names','artist_individual','artist_genre','track_name','country','region','language']
for col in cols_to_encode:
    spotify_data[col] = LabelEncoder().fit_transform(spotify_data[col])
#spotify_data_clean=spotify_data.dropna(subset=['valence','loudness','danceability','speechiness'])
#spotify_data_clean.reset_index(drop=True, inplace=True)
features=['region','language','valence','loudness','danceability','speechiness','artist_genre']
#encode_features=features=['language','artist_genre']
target='streams'
missing_values = spotify_data[['region','language','valence','loudness','danceability','speechiness','artist_genre', 'streams']].isnull().sum()
print(missing_values)
X=spotify_data[features]
y=spotify_data[target]
#X=spotify_data_clean[features]
#y=spotify_data_clean[target]
#X=pd.get_dummies(X, columns=['region','language','artist_genre'])
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

# plotting the output
plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions, color='blue', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)],[min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Linear Regression)')
plt.legend()
plt.show()
