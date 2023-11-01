# exercise 2.1.1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend

# Read the CSV and create a pd df
filename = '../Data/SpotifyDataSet.csv'
df = pd.read_csv(filename)

# Reduce df to only the observations where rank <= 50, only considering top 50 songs
df = df[df['rank'] <= 50]

# Extract class names for 'region' attribute and encode it
raw_data = df.values
classLabels = raw_data[:,-2]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))

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
classification_cols = ['language','valence','energy','loudness','speechiness','danceability','artist_genre']
df = df[classification_cols]


# ------------------ DATA TRANSFORMATION -------------------
# We need to do some transformation for categorical attributes to numerical values.
# Since we have to do several encodings and lots of values
#   the easiest way is to assign a numerical value to each unique value.

columns_encoded = {}
# For each attribute we want to encode:
for col in classification_cols:
    # Get unique values for each attribute
    unique_values = df[col].unique()
    # We need to do a dictionary
    encoding = {}
    
    # Assign unique value 'i' to 'attr_value' value in the possible values
    for i, attr_value in enumerate(unique_values):
        encoding[attr_value] = i
    
    columns_encoded[col] = encoding
    
    # Apply encoding to original dataframe with map function
    df[col] = df[col].map(encoding)

# ------------ DATA STANDARDIZATION -----------
# Standardize the numerical attributes, we can just divide the data by the std deviation
for col in classification_cols:
    std_dev = np.std(df[col], axis=0)
    df[col] = df[col]/std_dev

raw_data = df.values
# Now we can build the standardized matrix X and convert all values to float
cols = range(0,len(df.columns))
X = raw_data[:, cols]
X = X.astype(float)
N, M = X.shape
C = len(classNames)
# print(X)

# Extracting vector y based on the 'region' attribute
y = np.array([classDict[cl] for cl in classLabels])

# ------------------- PCA ANALYSIS -----------------------

# Let's subtract the mean value from data to center the data
Y = X - np.ones((N,1))*X.mean(axis=0)

# U: left singular vectors
# S: singular values (eigenvalues)
# V: right singular matrix
U,S,V = svd(Y,full_matrices=False)

# variation explained by principal components
rho = (S*S) / (S*S).sum() 

# we wish to get principal components that explain at least 90% of the variance
threshold = 0.9

# Plot for explained variance
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
# plt.show()
# From the previous plot, we conclude that we need at least 5 PCs



U,S,Vh = svd(Y,full_matrices=False)
V = Vh.T
Z = Y @ V
# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('Spotify Data: PCA components for the Region classes')
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))
# show()


# Let's get all the attribute names
attributeNames = np.asarray(df.columns[cols])

# Since the first 5 components explained 90% of the variance,
#   lets review the coefficients for the first 6.
pcs = [0,1,2,3,4,5]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)

# Plot for each PCA component
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Spotify DataSet: PCA Component Coefficients')
# plt.show()

# Let's print the components' coefficients
for i in range(0,6):
    print(f'PC{i+1}: ', V[:,i].T)

# Let's check the first observation for 'Europe'
europe_data = Y[y==9,:]
print(europe_data)
print(europe_data[0,:])

# Projections onto each principal component
for i in range(0,6):
    print(f'This observation projection onto PC{i}')
    print(europe_data[0,:]@V[:,i])