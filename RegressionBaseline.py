from numba import jit, cuda
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
import torch
from toolbox_02450 import train_neural_net, draw_neural_net
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

# Make figure for holding summaries (errors and learning curves)
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']


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


# ----------------Parameters for ANN--------------------
n_hidden_units = 4 # This is S in 2 level cross validation algorithm
# Calculating the loss with Mean Squared Error
loss_fn = torch.nn.MSELoss()
n_replicates = 1
max_iter = 12000
outer_fold_errors_ANN = []

optimal_hidden_units_array = []

for k1,(train_outer_index, test_outer_index) in enumerate(CV_outer.split(X,y)):
    print('Computing CV outer fold: {0}/{1}..'.format(k1+1,K_outer))

# -------- Setting BASELINE MODEL ------------
    # Outer cross validation loop. First make the outer split into K1 folds
    X_train_outer = X[train_outer_index,:]
    y_train_outer = y[train_outer_index]
    X_test_outer = X[test_outer_index,:]
    y_test_outer = y[test_outer_index]

# -------- Setting ANN MODEL ------------
    # Outer cross validation loop. First make the outer split into K1 folds
    X_train_outer_ANN = torch.Tensor(X_train_outer)
    y_train_outer_ANN = torch.Tensor(y_train_outer)
    X_test_outer_ANN = torch.Tensor(X_test_outer)
    y_test_outer_ANN = torch.Tensor(y_test_outer)

    # Initializing values for inner folds
    inner_validation_errors_ANN = []

# START OF INNER FOLDS
    for k2, (train_inner_index,test_inner_index) in enumerate(CV_inner.split(X_train_outer,y_train_outer)):
        print('\tComputing CV inner fold: {0}/{1}..'.format(k2+1,K_inner))
        # Inner cross validation loop. Use cross-validation to select optimal model.
        X_train_inner_ANN = torch.Tensor(X[train_inner_index,:])
        y_train_inner_ANN = torch.Tensor(y[train_inner_index])
        X_test_inner_ANN = torch.Tensor(X[test_inner_index,:])
        y_test_inner_ANN = torch.Tensor(y[test_inner_index])

        # Initializing array that contains error for each model inside this k2 fold
        validation_errors = []
        s_values_array = []
        # for each number of hidden units
        for s in range(1,n_hidden_units+1):
            # print("S: ",s)
            # Setting up Sequential model
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, s), #M features to H hidden units
                torch.nn.Tanh(),   
                torch.nn.Linear(s, 1), # H hidden units to 1 output neuron
            )

            # Training neural network on inner data
            net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train_inner_ANN,
                                                       y=y_train_inner_ANN,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)


            # Get predictions from ANN
            y_predicted_inner_ANN = net(X_test_inner_ANN)

            # Calculate validation error of model S and append to array of all validation errors
            model_validation_error = (y_predicted_inner_ANN.float()-y_test_inner_ANN.float())**2
            model_validation_error_rate = (sum(model_validation_error).type(torch.float)/len(y_test_inner_ANN)).data.numpy()[0]
            validation_errors.append(model_validation_error_rate)
            s_values_array.append(s)
        print(s_values_array)
        
        # Now we need to add the validation errors for this fold to an array containing errors for each inner fold
        inner_validation_errors_ANN.append(validation_errors)



# TRAINING FOR OUTER FOLDS

# -------- BASELINE MODEL ---------
    # Compute squared error without using the input data at all
    error_outer_train[k1] = np.square(np.mean(y_train_outer)-y_train_outer).sum()/y_train_outer.shape[0]
    error_outer_test[k1] = np.square(np.mean(y_test_outer)-y_test_outer).sum()/y_test_outer.shape[0]

# -------- ANN MODEL -------------
    # Performances for each model
    model_gen_errors_ANN_per_inner_fold = [np.array([inner_validation_errors_ANN[i][j] for i in range(K_inner)]) for j in range(len(validation_errors))]
    # print((model_gen_errors_ANN_per_inner_fold))
    estimated_inner_gen_error_ANN = []
    # Now, for each model S, compute inner generalization error
    for s in range(0,len(model_gen_errors_ANN_per_inner_fold)):
        s_inner_error = np.sum(np.multiply(len(y_test_inner_ANN),model_gen_errors_ANN_per_inner_fold[s])) / len(y_train_outer_ANN)
        estimated_inner_gen_error_ANN.append(s_inner_error)
        # print(s)

    # Select optimal model:
    optimal_estimated_inner_gen_error_ANN = min(estimated_inner_gen_error_ANN)
    optimal_number_hidden_units = s_values_array[estimated_inner_gen_error_ANN.index(optimal_estimated_inner_gen_error_ANN)]
    print(optimal_number_hidden_units)
    optimal_hidden_units_array.append(optimal_number_hidden_units)

    # Setting up Sequential model
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, optimal_number_hidden_units), #M features to H hidden units
        torch.nn.Tanh(),   
        torch.nn.Linear(optimal_number_hidden_units, 1), # H hidden units to 1 output neuron
    )

    # Training neural network on training outer data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train_outer_ANN,
                                                       y=y_train_outer_ANN,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    # Get predictions from ANN
    y_predicted_outer_ANN = net(X_test_outer_ANN)

    # Calculate outer folds errors based on formula
    outer_errors = (y_predicted_outer_ANN.float()-y_test_outer_ANN.float())**2
    outer_error_rate = (sum(outer_errors).type(torch.float)/len(y_test_outer_ANN)).data.numpy()[0]
    outer_fold_errors_ANN.append(outer_error_rate)

    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k1])
    h.set_label('CV fold {0}'.format(k1+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')


print("BASELINE MODEL ERRORS: ",error_outer_test)
print("ANN MODEL ERRORS: ",outer_fold_errors_ANN)
print("ANN OPTIMAL NUMBER OF HIDDEN UNITS PER FOLD: ",optimal_hidden_units_array)

# BASELINE MODEL RESULTS
generalization_error_baseline_model = np.mean(error_outer_test)
print('Est gen error of baseline model: ' +str(round(generalization_error_baseline_model, ndigits=3))) 

# ANN MODEL RESULTS

    # Display the error rate across folds
summaries_axes[1].bar(np.arange(1, K_outer+1), np.squeeze(np.asarray(outer_fold_errors_ANN)), color=color_list)
summaries_axes[1].set_xlabel('Fold')
summaries_axes[1].set_xticks(np.arange(1, K_outer+1))
summaries_axes[1].set_ylabel('Error rate')
summaries_axes[1].set_title('Test misclassification rates')

print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [0,2]]
draw_neural_net(weights, biases, tf, attribute_names=regression_cols)

# Print the average classification error rate
print('\nANN Generalization error/average error rate: {0}'.format(round(np.mean(outer_fold_errors_ANN),4)))
