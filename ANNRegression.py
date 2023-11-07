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

# Make figure for holding summaries (errors and learning curves)
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

# ------------ DATA STANDARDIZATION -----------

X = df_class.values
# Standardize the feature matrix X
scaler_X = preprocessing.StandardScaler()
X = scaler_X.fit_transform(X)
N, M = X.shape

y = df[['streams']].values
print(y.dtype)
print(X.dtype)

# ---------------- STARTING K-Fold CV-------------------
K_outer = 10
K_inner = 10
CV_outer = sklearn.model_selection.KFold(n_splits=K_outer,shuffle=True)
CV_inner = sklearn.model_selection.KFold(n_splits=K_inner,shuffle=True)


# ----------------Parameters for ANN--------------------
n_hidden_units = 3 # This is S in 2 level cross validation algorithm
# Calculating the loss with Mean Squared Error
loss_fn = torch.nn.MSELoss()
n_replicates = 2
max_iter = 10000



# Initializing error array
outer_fold_errors = []


for k1,(train_outer_index, test_outer_index) in enumerate(CV_outer.split(X,y)):
    print('Computing CV outer fold: {0}/{1}..'.format(k1+1,K_outer))

    # Outer cross validation loop. First make the outer split into K1 folds
    X_train_outer = torch.Tensor(X[train_outer_index,:])
    y_train_outer = torch.Tensor(y[train_outer_index])
    X_test_outer = torch.Tensor(X[test_outer_index,:])
    y_test_outer = torch.Tensor(y[test_outer_index])

    # Initializing values for inner folds
    inner_validation_errors_ANN = []

    # Starting inner fold
    for k2, (train_inner_index,test_inner_index) in enumerate(CV_inner.split(X_train_outer,y_train_outer)):
        print('\tComputing CV inner fold: {0}/{1}..'.format(k2+1,K_inner))
        # Inner cross validation loop. Use cross-validation to select optimal model.
        X_train_inner = torch.Tensor(X[train_inner_index,:])
        y_train_inner = torch.Tensor(y[train_inner_index])
        X_test_inner = torch.Tensor(X[test_inner_index,:])
        y_test_inner = torch.Tensor(y[test_inner_index])

        # Initializing array that contains error for each model inside this k2 fold
        validation_errors = []

        # for each number of hidden units
        for s in range(1,n_hidden_units+1):

            # Setting up Sequential model
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, s), #M features to H hidden units
                torch.nn.Tanh(),   
                torch.nn.Linear(s, 1), # H hidden units to 1 output neuron
            )

            # Training neural network on inner data
            net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train_inner,
                                                       y=y_train_inner,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)


            # Get predictions from ANN
            y_predicted_inner_ANN = net(X_test_inner)

            # Calculate validation error of model S and append to array of all validation errors
            model_validation_error = (y_predicted_inner_ANN.float()-y_test_inner.float())**2
            model_validation_error_rate = (sum(model_validation_error).type(torch.float)/len(y_test_inner)).data.numpy()[0]
            validation_errors.append(model_validation_error_rate)
            
        
        # Now we need to add the validation errors for this fold to an array containing errors for each inner fold
        inner_validation_errors_ANN.append(validation_errors)

    # We now need to get an array with the performances for each model.
    model_gen_errors_ANN_per_inner_fold = [np.array([inner_validation_errors_ANN[i][j] for i in range(K_inner)]) for j in range(n_hidden_units)]
    print(model_gen_errors_ANN_per_inner_fold)

    estimated_inner_gen_error_ANN = []
    # Now, for each model S, compute inner generalization error
    for s in range(0,len(model_gen_errors_ANN_per_inner_fold)):
        s_inner_error = np.sum(np.multiply(len(y_test_inner),model_gen_errors_ANN_per_inner_fold[s])) / len(y_train_outer)
        estimated_inner_gen_error_ANN.append(s_inner_error)

    print("Model performances for each amount of Hidden units: ",estimated_inner_gen_error_ANN)
    # Select optimal model:
    optimal_estimated_inner_gen_error_ANN = min(estimated_inner_gen_error_ANN)
    optimal_number_hidden_units = estimated_inner_gen_error_ANN.index(optimal_estimated_inner_gen_error_ANN) + 1


    # Setting up Sequential model
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, optimal_number_hidden_units), #M features to H hidden units
        torch.nn.Tanh(),   
        torch.nn.Linear(optimal_number_hidden_units, 1), # H hidden units to 1 output neuron
    )



    # --------------- Continuing with outer fold ----------------
    # Training neural network on training outer data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train_outer,
                                                       y=y_train_outer,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)

    # print('\nBest loss: ',final_loss)
    
    # Get predictions from ANN
    y_predicted_outer_ANN = net(X_test_outer)

    # Calculate outer folds errors based on formula
    outer_errors = (y_predicted_outer_ANN.float()-y_test_outer.float())**2
    outer_error_rate = (sum(outer_errors).type(torch.float)/len(y_test_outer)).data.numpy()[0]
    outer_fold_errors.append(outer_error_rate)
    
    print(outer_fold_errors)

    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k1])
    h.set_label('CV fold {0}'.format(k1+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')

# Display the error rate across folds
summaries_axes[1].bar(np.arange(1, K_outer+1), np.squeeze(np.asarray(outer_fold_errors)), color=color_list)
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
print('\nGeneralization error/average error rate: {0}%'.format(round(100*np.mean(outer_fold_errors),4)))

# print(test_error_baseline)