import numpy as np

class GaussianNB:
    def __init__(self, priors =None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing
        
    def logprior(self, class_ind):
        return np.log(self.class_priors_[class_ind])
    
    def loglikelihood(self, Xi, class_ind):
        """ P(x|c) - Likelihood """
        # mu: mean , var : variance , Xi: sample (a row of X)
        # Get the class mean
        mu = self.theta_[class_ind]
        # Get the class variance
        var = self.var_[class_ind]
        # Write the Gaussian Likelihood expression
        numerator = np.exp((-1/2)*((Xi-mu)**2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        GaussLikelihood = numerator / denominator
        # Take the log of GaussLikelihood
        logGaussLikelihood = np.log(GaussLikelihood)
        # Return loglikelihood of the sample . Now you will use the " naive "
        # part of the naive bayes .
        return np.sum(logGaussLikelihood)
    
    def posterior(self, Xi, class_ind):
        logprior = self.logprior(class_ind)
        loglikelihood = self.loglikelihood(Xi, class_ind)
        posterior = logprior + loglikelihood
        # Return posterior
        return posterior
    
    def fit(self, X, y):
        # Number of samples , number of features
        n_samples, n_features = X.shape
        # Get the unique classes
        self.classes_ = np.unique(y)
        # Number of classes
        n_classes = len(self.classes_)
        # Initialize attributes for each class
        # Feature means for each class , shape ( n_classes , n_features )
        self.theta_ = np.zeros((n_classes, n_features))
        # Feature variances for each class shape ( n_classes , n_features )
        self.var_ = np.zeros((n_classes, n_features))
        # Class priors shape ( n_classes ,)
        self.class_priors_ = np.zeros(n_classes)
        """ P(c) - Prior Class Probability """
        # Calculate class means , variances and priors
        for c_ind, c_id in enumerate(self.classes_):
            # Get the samples that belong to class c_id
            X_class = X[y == c_id]
            # Mean of the each feature that belongs to class c_id
            self.theta_ [c_ind, :] = np.mean(X_class, axis =0)
            # Calculate the variance of each feature that belongs to c_id
            self.var_[c_ind, :] = np.var(X_class, axis=0) + self.var_smoothing
            # Calculate the priors for each class
            self.class_priors_[c_ind] = len(X_class) / len(X)
            
    def predict(self, X):
        """ Calculates Posterior probability P(c|x) """
        y_pred = []
        for Xi in X: # Calculate posteriors for each sample
            posteriors = [] # For saving posterior values for each class
            # Calculate posterior probability for each class
            for class_ind in self.classes_:
                # Calculate posterior
                sample_posterior = self.posterior(Xi, class_ind)
                # Append the posterior value of this class to posteriors
                posteriors.append(sample_posterior)
                # Get the class that has the highest posterior prob . and
                # append the prediction for this sample to y_pred
            y_pred.append(self.classes_[np.argmax(posteriors)])
        # Return predictions for all samples
        return np.array(y_pred)
    
# GAUSSIAN NAIVE BAYES APPLICATION - cancer data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from naive_bayes import GaussianNB

# Load breast cancer data using load_breast_cancer and inspect it.
cancer = load_breast_cancer()
cancer.keys()
cancer.data
cancer.feature_names
cancer.target
cancer.target_names

df = pd.DataFrame(
    np.c_[cancer.data, cancer.target], 
    columns = [list(cancer.feature_names)+ ['target']]
                 )
df.head()
df.describe()

# Get the data and target
X = cancer.data
y = cancer.target
X.shape, y.shape
# X = df.drop('target', axis=1)  # Features
# y = df['target']  # Target

f,ax=plt.subplots(figsize=(15,15))
sns.heatmap(df.corr(),annot=True,linecolor="blue",fmt=".2f",ax=ax)
plt.show()

X.shape
# Get feature names
feature_names = cancer.feature_names
print(feature_names)
# Create a pandas dataframe
df_cancer = pd.DataFrame(np.c_[cancer.data, cancer.target], columns = [list(cancer.feature_names)+ ['target']])
# Compute pairwise correlation of features .
# See https :// pandas . pydata . org / docs / reference / api/ pandas . DataFrame . corr . html
corr = df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Pairwise Correlation Heatmap')
plt.show()
# Bayesian approach aims to obtain sufficient approximation with less data when compared with
# Frequentist approach. In Bayesian, with fixed data and random parameters, main purpose is to 
# update P(theta) with less of data. Therefore, we can make selection of features to reduce the data with
# parameters explain the data best.
# Selection of features based on 
# box plots and corresponding features scatter plots (given in end of this section of codes):
columns = list(cancer.feature_names)
column_indices = [columns.index('mean radius'), columns.index('mean perimeter'),
                 columns.index('mean compactness'), columns.index('worst concave points')]
X = X[:, column_indices]

# Split the data using train_test_split
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.4, random_state = 999)

clf = GaussianNB()
clf.fit(X_train , y_train)
predictions = clf.predict(X_test)
print("Accuracy score :", accuracy_score(y_test , predictions ))

results = pd.DataFrame({
    'Predictions': predictions,
    'Actual': y_test.flatten()  # Flatten y_test to make it 1D
})

print(results)
