#!/usr/bin/env python
# coding: utf-8

# In our project, we are embarking on an intriguing classification task where the primary goal is to predict the regional popularity of songs based on a range of distinct song attributes. These attributes include Language, Valence, Energy, Genre, and Danceability. This classification endeavor falls into the category of a multi-class problem, as we are not merely distinguishing between two outcomes, but rather classifying songs into multiple regional categories.
# 
# Our analysis seeks to unravel the complex tapestry of musical preferences across different regions. By leveraging these specific song attributes, we aim to forecast the region where a song is most likely to top the charts. This goes beyond mere prediction; it's an exploration into the nuanced musical tastes prevalent in various regions. We hypothesize that certain regions may demonstrate a marked preference for songs with particular characteristics – for instance, a region might favor high-energy tracks, while another might gravitate towards songs with a specific language or danceability factor.
# 
# By successfully classifying songs into their most likely popular regions, we can gain valuable insights into regional musical trends and preferences. This could not only aid in understanding cultural inclinations but also serve practical purposes, such as guiding artists and record labels in tailoring their releases to suit specific regional tastes, thereby optimizing their chances of success in those markets.

# In[27]:


import pandas as pd

# Loading the dataset
file_path = "C:/Users/lydi_/OneDrive/Documents/DTU master , lectures and exercises/Introduction to Machine Learning and Data Mining/Projects/Project 2/SpotifyDataSet.csv"
spotify_data = pd.read_csv(file_path)

# Displaying the first few rows of the dataset to understand its structure
spotify_data.head()


# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression




#keep only the columns that we want and also the response variable

# Selecting relevant features and the target variable
features = ['region','language', 'valence', 'energy', 'artist_genre', 'danceability']
spotify_data_selected=spotify_data[features]

#display the first rows of the dataframe
spotify_data_selected.head()


# In[29]:


spotify_data_selected.describe()


# In[30]:


spotify_data_selected.shape


# In[31]:


null_values=spotify_data_selected.isnull().sum()
null_values


# In[71]:


#feature matrix
X=spotify_data_selected.drop('region',axis=1)
#response variable
y=spotify_data_selected['region']

#In pandas, when you're using the `.drop()` method, the `axis` parameter specifies
#whether you're dropping labels from the index (`axis=0`) or columns (`axis=1`).
#- `axis=0`: This is the default and it means the operation should occur on the index, which would drop rows from the DataFrame.
#- `axis=1`: This means the operation should occur on the columns, which would drop columns from the DataFrame.
X.head()


# import numpy as np
# 
# # Assuming X is your feature DataFrame
# 
# # Convert 'language' and 'artist_genre' to integer codes
# language_codes = pd.factorize(X['language'])[0]
# artist_genre_codes = pd.factorize(X['artist_genre'])[0]
# 
# # One-hot encode 'language'
# K_language = language_codes.max() + 1
# language_encoding = np.zeros((len(language_codes), K_language))
# language_encoding[np.arange(len(language_codes)), language_codes] = 1
# 
# # One-hot encode 'artist_genre'
# K_genre = artist_genre_codes.max() + 1
# genre_encoding = np.zeros((len(artist_genre_codes), K_genre))
# genre_encoding[np.arange(len(artist_genre_codes)), artist_genre_codes] = 1
# 
# # Remove the original categorical columns
# X = X.drop(['language', 'artist_genre'], axis=1)
# 
# # Concatenate the encoded columns with the rest of your data
# X_encoded = np.concatenate([X.values, language_encoding, genre_encoding], axis=1)
# 

# To include qualitative attributes (categorical variables) in a regression model, we need to convert them into a format that the model can understand. This is typically done through one-hot encoding also known as dummy coding. 

# In[53]:


pip install category_encoders


# In[73]:


import category_encoders as ce
encoder = ce.OneHotEncoder(cols=['language', 'artist_genre'], use_cat_names=True, drop_invariant=True)
X_encoded = encoder.fit_transform(X)
print(X_encoded.head())


# In[75]:


X=X_encoded


# In[89]:


from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, ylim, show
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split

#split the data into Training and testing data
#purpose: to evaluate the model's performance on unseen data
#method:use the train_test_split from sklearn.model_selection

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#fit logistic regression model
model = lm.LogisticRegression(max_iter=1000)
model=model.fit(X_train,y_train)


# In[90]:


#predict class labels and probabilitys 
y_est=model.predict(X_test) #y_est will contain the predictes class labels
y_est_prob=model.predict_proba(X_test) # will be a 2D array where each row 
#corresponds to a sample, and each column corresponds to the probability of that sample belonging to one of the class


# In[93]:


#calculate the accuracy of the model which is the proportion of the correct predctions

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_est)
print(f"Accuracy: {accuracy:.2f}")


# In[95]:


#confusion matrix 
#the confusion matrix shows the correct and incorrrect predictions for each class

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_est)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[ ]:


from sklearn.model_selection import cross_val_score

cross_val_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validated Accuracy: {np.mean(cross_val_accuracy):.2f}")

