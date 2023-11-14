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
from sklearn.neighbors import KNeighborsClassifier

from toolbox_02450.statistics import correlated_ttest
pd.options.mode.chained_assignment = None  # default='warn'

# Read the CSV and create a pd df
filename = 'SpotifyDataSet.csv'
df = pd.read_csv(filename)

# We want to encode all categorical attributes
#    since there are so many possible values, we'll use the label encoder
#    function from sklearn.preprocessing
class_mappings = {}
# print(df['region'].unique())
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




#------------------- Define the quadratic loss function   -----------------
def quadratic_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


#------------------ Initialize arrays to store quadratic losses for each model ------------
quadratic_loss_baseline = []
quadratic_loss_knn = []
quadratic_loss_logistic = []


# ------------ DATA STANDARDIZATION -----------

X = df_class.values
# Standardize the feature matrix X
scaler_X = preprocessing.StandardScaler()
X = scaler_X.fit_transform(X)
N, M = X.shape

y = df[['region']].values
# print(y)

# ---------------- STARTING K-Fold CV-------------------
K_outer = 10
K_inner = 10
CV_outer = sklearn.model_selection.KFold(n_splits=K_outer,shuffle=True)
CV_inner = sklearn.model_selection.KFold(n_splits=K_inner,shuffle=True)

# ------------------- BASELINE PARAMETERS ---------------------
error_outer_train = np.empty((K_outer,1))
error_outer_test = np.empty((K_outer,1))
test_error_baseline = np.empty((K_outer,1))


# ------------------- KNN PARAMETERS ---------------------
# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
knn_range = np.array(range(5,11))
dist=2
metric = 'minkowski'
metric_params = {} # no parameters needed for minkowski
knn_outer_fold_errors = []
optimal_knn_numbers_array = []


# ------------------- LOGISTIC PARAMETERS --------------------
lambda_interval = np.logspace(0, 4, 20)
logistic_outer_fold_errors = []
optimal_logistic_numbers_array = []

# STARTING OUTER FOLDS
for k1,(train_outer_index, test_outer_index) in enumerate(CV_outer.split(X,y)):
    print('Computing CV outer fold: {0}/{1}..'.format(k1+1,K_outer))

    # extract training and test set for current CV fold
    X_train_outer = X[train_outer_index,:]
    y_train_outer = y[train_outer_index].flatten()
    X_test_outer = X[test_outer_index,:]
    y_test_outer = y[test_outer_index].flatten()
    internal_cross_validation = 10

    error_inner_train = np.empty((K_inner,1))
    error_inner_test = np.empty((K_inner,1))

    knn_inner_validation_errors = []
    logistic_inner_validation_errors = []

# START OF INNER FOLDS
    for k2, (train_inner_index,test_inner_index) in enumerate(CV_inner.split(X_train_outer,y_train_outer)):
        print('\tComputing CV inner fold: {0}/{1}..'.format(k2+1,K_inner))

        # ------------ KNN INNER FOLDS -------------
        # Inner cross validation loop. Use cross-validation to select optimal model.
        X_train_inner_KNN = X[train_inner_index,:]
        y_train_inner_KNN = y[train_inner_index].flatten()
        X_test_inner_KNN = X[test_inner_index,:]
        y_test_inner_KNN = y[test_inner_index].flatten()

        # Initializing array that contains error for each model inside this k2 fold
        knn_model_errors = []
        s_values_array = []
        # for each number of knn neigbors
        for s in knn_range:

            # Fit classifier and classify the test points
            knclassifier = KNeighborsClassifier(n_neighbors=s, p=dist, 
                                    metric=metric,
                                    metric_params=metric_params)


            knclassifier.fit(X_train_inner_KNN, y_train_inner_KNN)
            y_est_inner_KNN = knclassifier.predict(X_test_inner_KNN)

            knn_error_per_model = np.sum(y_est_inner_KNN != y_test_inner_KNN)/len(y_test_inner_KNN)
            knn_model_errors.append(knn_error_per_model)
            s_values_array.append(s)
            # Now we need to add the validation errors for this fold to an array containing errors for each inner fold
        # print(s_values_array)
        knn_inner_validation_errors.append(knn_model_errors)

        # ----------- LOGISTIC REGRESSION INNER FOLDS --------------

        X_train_inner_LOG = X[train_inner_index,:]
        y_train_inner_LOG = y[train_inner_index].flatten()
        X_test_inner_LOG = X[test_inner_index,:]
        y_test_inner_LOG = y[test_inner_index].flatten()

        # Initializing array that contains error for each model inside this k2 fold
        logistic_model_errors = []
        log_s_values_array = []
        # for each number of lambda
        for s in lambda_interval:

            # Fit classifier and classify the test points
            mdl = lm.LogisticRegression(penalty='l2', C=1/s, max_iter=1000, multi_class='multinomial', solver='lbfgs')
            mdl.fit(X_train_inner_LOG, y_train_inner_LOG)
            y_est_inner_LOG = mdl.predict(X_test_inner_LOG)

            logistic_error_per_model = np.sum(y_est_inner_LOG != y_test_inner_LOG)/len(y_test_inner_LOG)
            logistic_model_errors.append(logistic_error_per_model)
            log_s_values_array.append(s)
            # Now we need to add the validation errors for this fold to an array containing errors for each inner fold
        # print(log_s_values_array)
        logistic_inner_validation_errors.append(logistic_model_errors)



# ------------------- OUTER FOLD KNN MODEL -------------------
    # Performances for each model
    errors_KNN_model_per_fold = [np.array([knn_inner_validation_errors[i][j] for i in range(K_inner)]) for j in range(len(knn_model_errors))]

    estimated_inner_gen_error_KNN = []
    # Now, for each model S, compute inner generalization error
    for s in range(0,len(errors_KNN_model_per_fold)):
        s_inner_error = np.sum(np.multiply(len(y_test_inner_KNN),errors_KNN_model_per_fold[s])) / len(y_train_outer)
        estimated_inner_gen_error_KNN.append(s_inner_error)
    # print(estimated_inner_gen_error_KNN)
    # Select optimal model:
    optimal_estimated_inner_gen_error_KNN = min(estimated_inner_gen_error_KNN)
    optimal_knn_number = s_values_array[estimated_inner_gen_error_KNN.index(optimal_estimated_inner_gen_error_KNN)]

    optimal_knn_numbers_array.append(optimal_knn_number)
    # print(optimal_knn_number)
    # Fit classifier and classify the test points
    knclassifier = KNeighborsClassifier(n_neighbors=optimal_knn_number, p=dist, 
                            metric=metric,
                            metric_params=metric_params)

    knclassifier.fit(X_train_outer, y_train_outer)
    y_est_outer_KNN = knclassifier.predict(X_test_outer)

    knn_error_per_model = 100*np.sum(y_est_outer_KNN != y_test_outer)/len(y_test_outer)
    knn_outer_fold_errors.append(knn_error_per_model)
    
    
    
    loss_outer_knn = quadratic_loss(y_test_outer, y_est_outer_KNN)
    quadratic_loss_knn.append(loss_outer_knn)


# ------------------- OUTER FOLD LOGISTIC MODEL --------------------
    # Performances for each model
    errors_LOG_model_per_fold = [np.array([logistic_inner_validation_errors[i][j] for i in range(K_inner)]) for j in range(len(logistic_model_errors))]

    estimated_inner_gen_error_LOG = []
    # Now, for each model S, compute inner generalization error
    for s in range(0,len(errors_LOG_model_per_fold)):
        s_inner_error = np.sum(np.multiply(len(y_test_inner_LOG),errors_LOG_model_per_fold[s])) / len(y_train_outer)
        estimated_inner_gen_error_LOG.append(s_inner_error)
    # print(estimated_inner_gen_error_LOG)
    # Select optimal model:
    optimal_estimated_inner_gen_error_LOG = min(estimated_inner_gen_error_LOG)
    optimal_logistic_number = log_s_values_array[estimated_inner_gen_error_LOG.index(optimal_estimated_inner_gen_error_LOG)]

    optimal_logistic_numbers_array.append(optimal_logistic_number)
    # print(optimal_logistic_number)
    # Fit classifier and classify the test points
    mdl = lm.LogisticRegression(penalty='l2', C=1/optimal_logistic_number, max_iter=1000, multi_class='multinomial', solver='lbfgs')
    mdl.fit(X_train_inner_LOG, y_train_inner_LOG)
    y_est_outer_LOG = mdl.predict(X_test_outer)

    logistic_error_per_model = 100*np.sum(y_est_outer_LOG != y_test_outer)/len(y_test_outer)
    logistic_outer_fold_errors.append(logistic_error_per_model)
    
    
    
    loss_outer_LOG = quadratic_loss(y_test_outer, y_est_outer_LOG)
    quadratic_loss_logistic.append(loss_outer_LOG)


# ------------------- OUTER FOLD BASELINE MODEL --------------------
    # print(y_test_outer)
    # Choosing most present class on the training data
    best_class = np.argmax(np.bincount(y_train_outer))
    # print(best_class)
    # Creating numpy array full of only best predicted class
    baseline_preds = best_class*np.ones((y_test_outer.shape[0],1))
    # print(baseline_preds.flatten())
    # Compare predicted output to test
    bool_error = baseline_preds != y_test_outer
    # print(bool_error)
    # Estimate error by comparing y_test to largest class
    test_error_baseline[k1]=(100*np.sum(baseline_preds.flatten() != y_test_outer)/float(len(y_test_outer)))
    
    
    
    loss_outer_baseline = quadratic_loss(y_test_outer, baseline_preds.flatten())
    quadratic_loss_baseline.append(loss_outer_baseline)
    
    

#------------------statistical evaluation----------------------------------
quadratic_loss_baseline = np.array(quadratic_loss_baseline)
quadratic_loss_knn = np.array(quadratic_loss_knn)
quadratic_loss_LOG = np.array(quadratic_loss_logistic)    


# Perform the statistical evaluation (Setup II)
p_value_baseline_vs_knn, CI_baseline_vs_knn = correlated_ttest((quadratic_loss_baseline - quadratic_loss_knn), rho=1/K_outer,alpha=0.05)
p_value_baseline_vs_logistic, CI_baseline_vs_logistic = correlated_ttest(quadratic_loss_baseline - quadratic_loss_LOG, rho=1/K_outer,alpha=0.05)
p_value_knn_vs_logistic, CI_knn_vs_logistic = correlated_ttest(quadratic_loss_knn - quadratic_loss_LOG, rho=1/K_outer,alpha=0.05)


print("BASELINE TEST ERRORS PER FOLD: ",test_error_baseline)
print("KNN OPTIMAL NUMBERS PER FOLD: ", optimal_knn_numbers_array)
print("KNN ERRORS: ",knn_outer_fold_errors)
print("LOG OPTIMAL NUMBERS PER FOLD: ", optimal_logistic_numbers_array)
print("LOG ERRORS: ",logistic_outer_fold_errors)


print('----------------------- RESULTS -----------------------')
print('Fold    Logistic Regression       K NN              Baseline')
print('           l      Etest         k_num   Etest        Etest')
for i in range(0,k1):
    resa = "  " + str(i) + "       " + str(optimal_logistic_numbers_array[i]) + "     " + str(round(logistic_outer_fold_errors[i],2)) + "       " + str(optimal_knn_numbers_array[i] ) + "    " + str(round(knn_outer_fold_errors[i],2)) + "        " + str(round(test_error_baseline.flatten()[i],2))
    print(resa)

# BASELINE MODEL RESULTS
print("\n--------------------BASELINE MODEL--------------------")
generalization_error_baseline_model = np.mean(test_error_baseline)
print('\n-Estimated generalization error for baseline model: ',round(generalization_error_baseline_model, ndigits=2))

# KNN MODEL RESULTS
print("\n--------------------KNN MODEL--------------------")
generalization_error_knn_model = np.mean(knn_outer_fold_errors)
print('\n-Estimated generalization error for KNN model: ' ,round(generalization_error_knn_model, ndigits=2))

# LOGISTIC REGRESSION MODEL RESULTS
print("\n--------------------KNN MODEL--------------------")
generalization_error_logistic_model = np.mean(logistic_outer_fold_errors)
print('\n-Estimated generalization error for LOG model: ' ,round(generalization_error_logistic_model, ndigits=2))


# generalization_error_KNN_model = np.sum(np.multiply(knn_outer_fold_errors,data_outer_test_length)) * (1/N)
# print('est gen error of KNN model: ' +str(round(generalization_error_KNN_model, ndigits=3)))


# Print the results
print("\nBaseline vs KNN: p-value =", p_value_baseline_vs_knn, "CI =", CI_baseline_vs_knn)
print("Baseline vs Logistic Regression: p-value =", p_value_baseline_vs_logistic, "CI =", CI_baseline_vs_logistic)
print("KNN vs Logistic Regression: p-value =", p_value_knn_vs_logistic, "CI =", CI_knn_vs_logistic)

