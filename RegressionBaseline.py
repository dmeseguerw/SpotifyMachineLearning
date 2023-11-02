from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
import sklearn
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
from sklearn.calibration import LabelEncoder
import sklearn.linear_model as lm
from sklearn import model_selection
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

# Read the CSV and create a pd df
filename = 'SpotifyDataSet.csv'
df = pd.read_csv(filename)

# We want to encode all categorical attributes
#    since there are so many possible values, we'll use the label encoder
#    function from sklearn.preprocessing

cols_to_encode = ['week','artist_names','artist_individual','artist_genre','track_name','country','region','language']
for col in cols_to_encode:
    df[col] = LabelEncoder().fit_transform(df[col])


# We'll reduce the df to only the attributes mentioned for the classification task
classification_cols = ['language','valence','energy','danceability','artist_genre']
df_class = df[classification_cols]


# Attributes for regression
regression_cols = ["artist_genre","language","region","loudness","speechiness","danceability","valence"]
df_regression = df[regression_cols]



# ------------ DATA STANDARDIZATION -----------

X = df_regression.values
# Standardize the feature matrix X
scaler_X = preprocessing.StandardScaler()
X = scaler_X.fit_transform(X)
N, M = X.shape

y = df[['streams']].values

# ---------------- STARTING K-Fold CV-------------------
K_outer = 10
K_inner = 10
CV_outer = sklearn.model_selection.KFold(n_splits=K_outer,shuffle=True)
CV_inner = sklearn.model_selection.KFold(n_splits=K_inner,shuffle=True)
# Initialize variable
error_outer_train = np.empty((K_outer,1))
error_outer_test = np.empty((K_outer,1))

data_outer_test_length = []


k_out=0

for train_outer_index, test_outer_index in CV_outer.split(X):
    print('Computing CV outer fold: {0}/{1}..'.format(k_out+1,K_outer))

    # extract training and test set for current CV fold
    X_train_outer = X[train_outer_index,:]
    y_train_outer = y[train_outer_index]
    X_test_outer = X[test_outer_index,:]
    y_test_outer = y[test_outer_index]
    internal_cross_validation = 10

    error_inner_train = np.empty((K_inner,1))
    error_inner_test = np.empty((K_inner,1))

    data_outer_test_length.append(float(len(y_test_outer)))

    k_in = 0
    for train_inner_index, test_inner_index in CV_inner.split(X):
        print('Computing CV inner fold: {0}/{1}..'.format(k_out+1,K_inner))
        X_train_inner = X[train_inner_index,:]
        y_train_inner = y[train_inner_index]
        X_test_inner = X[test_inner_index,:]
        y_test_inner = y[test_inner_index]

        error_inner_train[k_in] = np.square(y_train_inner-y_train_inner.mean()).sum()/y_train_inner.shape[0]
        error_inner_test[k_in] = np.square(y_test_inner-y_test_inner.mean()).sum()/y_test_inner.shape[0]

        k_in+=1
    
    # Compute squared error without using the input data at all
    error_outer_train[k_out] = np.square(np.mean(y_train_outer)-y_train_outer).sum()/y_train_outer.shape[0]
    error_outer_test[k_out] = np.square(np.mean(y_test_outer)-y_test_outer).sum()/y_test_outer.shape[0]

    k_out+=1


print(error_inner_train)
print(error_inner_test)
print(error_outer_test)
print(error_outer_train)

## Estimate the generalization error
generalization_error_baseline_model = np.mean(error_outer_test)
print('est gen error of baseline model: ' +str(round(generalization_error_baseline_model, ndigits=3))) 