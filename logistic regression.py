#!/usr/bin/env python
# coding: utf-8

# In our project, we are embarking on an intriguing classification task where the primary goal is to predict the regional popularity of songs based on a range of distinct song attributes. These attributes include Language, Valence, Energy, Genre, and Danceability. This classification endeavor falls into the category of a multi-class problem, as we are not merely distinguishing between two outcomes, but rather classifying songs into multiple regional categories.
# 
# Our analysis seeks to unravel the complex tapestry of musical preferences across different regions. By leveraging these specific song attributes, we aim to forecast the region where a song is most likely to top the charts. This goes beyond mere prediction; it's an exploration into the nuanced musical tastes prevalent in various regions. We hypothesize that certain regions may demonstrate a marked preference for songs with particular characteristics – for instance, a region might favor high-energy tracks, while another might gravitate towards songs with a specific language or danceability factor.
# 
# By successfully classifying songs into their most likely popular regions, we can gain valuable insights into regional musical trends and preferences. This could not only aid in understanding cultural inclinations but also serve practical purposes, such as guiding artists and record labels in tailoring their releases to suit specific regional tastes, thereby optimizing their chances of success in those markets.

# In[12]:


import pandas as pd
import numpy as np
# Loading the dataset
file_path = "C:/Users/lydi_/OneDrive/Documents/DTU master , lectures and exercises/Introduction to Machine Learning and Data Mining/Projects/Project 2/SpotifyDataSet.csv"
spotify_data = pd.read_csv(file_path)

# Displaying the first few rows of the dataset to understand its structure
spotify_data.head()


# In[13]:


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


# In[14]:


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


# In[15]:


import category_encoders as ce
encoder = ce.OneHotEncoder(cols=['language', 'artist_genre'], use_cat_names=True, drop_invariant=True)
X_encoded = encoder.fit_transform(X)


# In[18]:


X=X_encoded


# In[20]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
import numpy as np

# Define a range for the regularization strength
lambda_interval = np.logspace(-8, 2, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))

# Train the model and evaluate for multi-class
for k in range(len(lambda_interval)):
    mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k], max_iter=1000, multi_class='multinomial', solver='lbfgs')
    mdl.fit(X_train_scaled, y_train)
    
    y_train_est = mdl.predict(X_train_scaled)
    y_test_est = mdl.predict(X_test_scaled)
    
    train_error_rate[k] = np.mean(y_train_est != y_train)
    test_error_rate[k] = np.mean(y_test_est != y_test)
    coefficient_norm[k] = np.sqrt(np.sum(mdl.coef_**2))

# Identify the optimal lambda
min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]

# Use the model with the optimal lambda for predictions
optimal_mdl = LogisticRegression(penalty='l2', C=1/opt_lambda, max_iter=10000, multi_class='multinomial', solver='saga')
optimal_mdl.fit(X_train_scaled, y_train)
y_test_pred = optimal_mdl.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))

