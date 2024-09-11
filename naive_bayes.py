import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from scipy import linalg
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from os import sys
import utils as ut

# Load dataset
title =  "binary_blobs"
path = "../datasets/toy/{}.csv".format(title)

# Get the data and its labels
X, y = ut.read_dataset(path, "label")

# Plot dataset
ut.plot_dataset(X, "f0", "f1", y, title) 

# Get unique labels
unique_labels = y.unique()
print("Unique Labels:", unique_labels)

# Classification model
clf = GaussianNB()

# Train the model with the data
clf.fit(X, y)

# Parameters that are learned from the data

# class_count_ attribute: Number of training samples observed in each class.
for label, count, in zip(unique_labels, clf.class_count_):
    print("{} training samples observed in class {}".format(count, label))
    
# class_prior_ attribute: Probability of each class
for label, prior_prob, in zip(unique_labels, clf.class_prior_):
    print("Probability of class {} is {}".format(label, prior_prob))

# theta_ attribute: Mean of each feature per class.
for label, mean, in zip(unique_labels, clf.theta_):
    print("Class {} means for each feature are {}".format(label, mean))

# var_ attribute: Variance of each feature per class.    
for label, variance, in zip(unique_labels, clf.var_):
    print("Class {} variances for each feature are {}".format(label, variance))  

# Names of features seen during fit. 
# Defined only when X has feature names that are all strings.    
for label, feature_names, in zip(unique_labels, clf.feature_names_in_):
    print("Class {} feature name is {}".format(label, feature_names))    


# Plot dataset
ut.plot_dataset(X, "f0", "f1", y, title) 

# Plot decision boundary
ut.plot_boundary(clf, X, "f0", "f1")

# Display means
ut.disp_point(clf.theta_)

means = clf.theta_
sigmas = clf.var_

conf_level = 2 # But remember our discussions about chi square distribution

# Plot ellipses for each class with "conf_level" confidence level
for mean, sigma in zip(means, sigmas):
    covar = [[sigma[0], 0.0],
             [0.0, sigma[1]]]
    eig_val, eig_vec = np.linalg.eigh(covar)    
    major_length = 2*conf_level*(np.sqrt(eig_val[0]))
    minor_length = 2*conf_level*(np.sqrt(eig_val[1]))

    
    u = eig_vec[0] / linalg.norm(eig_vec[0])
    angle = np.arctan(u[1]/u[0])
    angle = 180.0*angle/np.pi
    
    ellipse = mpl.patches.Ellipse(mean, major_length, minor_length, angle, 
                              linestyle="-", linewidth=2, 
                              edgecolor="black", facecolor='none', 
                              alpha=0.7, zorder=30)
    ax = plt.gca()
    ellipse.set_clip_box(ax.bbox)
    ax.add_artist(ellipse)