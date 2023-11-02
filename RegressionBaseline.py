# exercise 2.1.1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
import sklearn
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

# Read the CSV and create a pd df
filename = 'SpotifyDataSet.csv'
df = pd.read_csv(filename)

# Reduce df to only the observations where rank <= 50, only considering top 50 songs
df = df[df['rank'] <= 50]

# Extract class names for 'region' attribute and encode it
raw_data = df.values
streams_idx = df.columns.get_loc('streams')
# classLabels = raw_data[:,streams_idx]
# classNames = np.unique(classLabels)
# classDict = dict(zip(classNames,range(len(classNames))))

# We want to encode all categorical attributes
#    since there are so many possible values, we'll use the label encoder
#    function from sklearn.preprocessing

# cols_to_encode = ['week','artist_names','artist_individual','artist_genre','track_name','country','region','language']
# for col in cols_to_encode:
#     df[col] = LabelEncoder().fit_transform(df[col])


# We also want to fill the missing values with the median value:
cols_missing = ['danceability', 'energy', 'key', 'mode', 'loudness', 
                               'speechiness', 'acousticness', 'instrumentalness', 
                               'liveness', 'valence', 'tempo', 'duration']


df[cols_missing] = df[cols_missing].fillna(df[cols_missing].mean())


# We'll reduce the df to only the attributes mentioned for the classification task
classification_cols = ['language','valence','energy','danceability','artist_genre']
df_class = df[classification_cols]

# Attributes for regression
regression_cols = ["loudness","speechiness","danceability","valence"]
df_regression = df[regression_cols]


# ------------------ DATA TRANSFORMATION -------------------
# We need to do some transformation for categorical attributes to numerical values.
# Since we have to do several encodings and lots of values
#   the easiest way is to assign a numerical value to each unique value.

# columns_encoded = {}
# # For each attribute we want to encode:
# for col in regression_cols:
#     # Get unique values for each attribute
#     unique_values = df_regression[col].unique()
#     # We need to do a dictionary
#     encoding = {}
    
#     # Assign unique value 'i' to 'attr_value' value in the possible values
#     for i, attr_value in enumerate(unique_values):
#         encoding[attr_value] = i
    
#     columns_encoded[col] = encoding
    
#     # Apply encoding to original dataframe with map function
#     df_regression[col] = df_regression[col].map(encoding)

# ------------ DATA STANDARDIZATION -----------

raw_data = df_regression.values
# Now we can build the standardized matrix X and convert all values to float
cols = range(0,len(df_regression.columns))
X = df_regression.to_numpy(dtype=np.float32)
N, M = X.shape
Xt = X - np.ones((N,1))*X.mean(axis=0) # Subtract mean
Xt = Xt*(1/np.std(Xt,0)) # Divide by std deviation
X = Xt
# C = len(classNames)
# print(X)

# Extracting vector y based on the 'region' attribute
# y = np.array([classDict[cl] for cl in classLabels])

y = df['rank'].values
print(y)

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