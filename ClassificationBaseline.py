from scipy import stats
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
class_mappings = {}
print(df['region'].unique())
cols_to_encode = ['week','artist_names','artist_individual','artist_genre','track_name','country','region','language']
for col in cols_to_encode:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    if col == 'region':
        class_mappings[col] = {index: class_label for index, class_label in enumerate(encoder.classes_)}

# Print the class-to-number mapping
print(class_mappings)


# We'll reduce the df to only the attributes mentioned for the classification task
classification_cols = ['language','valence','energy','danceability','artist_genre']
df_class = df[classification_cols]


# Attributes for regression
regression_cols = ["artist_genre","language","region","loudness","speechiness","danceability","valence"]
df_regression = df[regression_cols]



# ------------ DATA STANDARDIZATION -----------

X = df_class.values
# Standardize the feature matrix X
scaler_X = preprocessing.StandardScaler()
X = scaler_X.fit_transform(X)
N, M = X.shape

y = df[['region']].values
print(y)

# ---------------- STARTING K-Fold CV-------------------
K_outer = 10
K_inner = 10
CV_outer = sklearn.model_selection.KFold(n_splits=K_outer,shuffle=True)
CV_inner = sklearn.model_selection.KFold(n_splits=K_inner,shuffle=True)
# Initialize variable
error_outer_train = np.empty((K_outer,1))
error_outer_test = np.empty((K_outer,1))
test_error_baseline = np.empty((K_outer,1))

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

    # Choosing largest class on the training data
    y_train = y_train_outer.flatten()
    y_test = y_test_outer.flatten()
    best_class = np.argmax(np.bincount(y_train))
    print(y_test_outer)
    baseline_preds = best_class*np.ones((y_test_outer.shape[0],1))
    # print(baseline_preds != y_test_outer)
    bool_error = baseline_preds != y_test_outer
    # bool_error = bool_error.flatten()
    print(np.sum(baseline_preds != y_test_outer))
    # Estimate error by comparing y_test to largest class
    test_error_baseline[k_out]=(100*np.sum(baseline_preds != y_test_outer)/float(len(y_test_outer)))
    class_counts = np.bincount(y_test_outer.flatten())
    print(f"Class counts in y_train for outer fold {k_out + 1}: {class_counts}")
    k_out+=1

print(test_error_baseline)