import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def read_dataset(path, label_column):

    dataset = pd.read_csv(path) 
    features = dataset.columns.tolist()
    features.remove(label_column)
    labels = dataset.loc[:, label_column] 
    data = dataset.loc[:, features] 
    
    return data, labels


def plot_dataset(data, col1, col2, labels, title=""):

    data = data.loc[:, [col1, col2]] 
    
    plt.figure()
    plt.title(title, size=20)

    for label in sorted(labels.unique()):
        label_points = data[labels==label].values
        x_points = label_points[:,0]
        y_points = label_points[:,1]
        label_text = "class-{}".format(label)
        plt.scatter(x_points, y_points, marker="o", alpha=0.70, 
                    label=label_text)
    
    plt.legend(loc="best", fontsize=14)
    plt.xlabel(col1, size=14)
    plt.ylabel(col2, rotation=0, size=14)



def disp_point(point, ax=None):
    if ax:
        ax.scatter(point[:,0], point[:,1], marker="o", color="black", s=60, 
                   alpha=0.7)
    else:
        plt.scatter(point[:,0], point[:,1], marker="o", color="black", s=40, 
                    alpha=0.5)
                    
                                                 
#def plot_boundary(model):

#    x_min, x_max = plt.xlim()
#    y_min, y_max = plt.ylim()
       
#    h = .05  
    
#    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#    grid_points = np.c_[xx.ravel(), yy.ravel()]
#    Z = model.predict(grid_points)
#    Z = Z.reshape(xx.shape)
#
#    plt.imshow(Z, interpolation="gaussian", zorder=-100, 
#               alpha=0.3, extent=[x_min, x_max, y_min, y_max], 
#               aspect="auto", origin="lower")  
               
               
def plot_boundary(model, data, col0, col1):

    mins = data[[col0, col1]].min()
    maxs = data[[col0, col1]].max()
    
    xmin, xmax = mins[0], maxs[0]
    ymin, ymax = mins[1], maxs[1]
    
    x_min = xmin - np.abs(xmax-xmin)*0.05
    x_max = xmax + np.abs(xmax-xmin)*0.05
    y_min = ymin - np.abs(ymax-ymin)*0.05
    y_max = ymax + np.abs(ymax-ymin)*0.05
    
    # determine bounds of meshgrid 
    #x_min, x_max = plt.xlim()
    #y_min, y_max = plt.ylim()    
    
    h = .05 
    
    _x, _y = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    grid_points = np.c_[_x.ravel(), _y.ravel()]
    
    grid_predictions = model.predict(grid_points)
    
    grid_predictions = grid_predictions.reshape(_x.shape)
    
    plt.imshow(grid_predictions, interpolation="gaussian", 
               alpha=0.3, extent=[x_min, x_max, y_min, y_max], 
               aspect="auto", origin="lower")
               
               

def plot_ellipses(model, factor=2.0):

    means  = model.theta_ 
    sigmas = model.var_ 
    
    for mean, sigma in zip(means, sigmas):
             
        covar = [[sigma[0], 0.0],
                 [0.0, sigma[1]]]
        
        eig_val, eig_vec = np.linalg.eigh(covar)
        

        majorLength = 2*factor*np.sqrt(eig_val[0])
        minorLength = 2*factor*np.sqrt(eig_val[1])


        u = eig_vec[0] / np.linalg.norm(eig_vec[0])
        angle = np.arctan(u[1]/u[0])
        angle = 180.0*angle/np.pi
        
        ellipse = Ellipse(mean, majorLength, minorLength, angle, 
                      linestyle="--", linewidth=2, 
                      edgecolor="black", facecolor='none', 
                      alpha=0.5, zorder=-30)
                      
        ax = plt.gca()
        ellipse.set_clip_box(ax.bbox)
        ax.add_artist(ellipse)      
 
 
def describe_dataset(X, y):
    print("Data shape: {}".format(X.shape))
    # how many labels for each class
    for label in np.unique(y):
        count = y[y==label].shape[0]
        print("{} instances for class {}".format(count, label))
    print("") 


def evaluate_model(model, true_labels, predicted_labels):
    acc_score = accuracy_score(true_labels, predicted_labels)
    print("Accuracy Score: {}".format(acc_score))
    report = classification_report(true_labels, predicted_labels)
    print("Classification Report: \n{}".format(report))
    print("") 
    
 
def test_model(model, path, label_column):

    # Read dataset
    X, y = read_dataset(path, label_column)
    
    # Describe data set
    print("Complete Data Set: ")
    describe_dataset(X, y)
    plot_dataset(X, "f0", "f1", y) #, "o", dataset_name)    
    xlim, ylim = plt.xlim(), plt.ylim()    
    
    # Split data set
    ratio = 0.3
    random_state = 22
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=ratio, 
                                                    random_state=random_state,
                                                    shuffle=True)
    print("Train Data: ")
    describe_dataset(X_train, y_train)

        
    print("Test Data: ")
    describe_dataset(X_test, y_test)
    
    # Fit model to TRAIN set
    model.fit(X_train, y_train)
    
    # Model performance on TRAIN set
    y_train_predicted = model.predict(X_train)
    print("Model Evaluation on TRAIN set")
    evaluate_model(model, y_train, y_train_predicted)
    
    plot_dataset(X_train, "f0", "f1", y_train) #, "o", title1)
    _ = plt.xlim(xlim), plt.ylim(ylim)
    plot_boundary(model) 
    plot_ellipses(model)    
    
    # Model performance on TEST set
    y_test_predicted = model.predict(X_test)
    print("Model Evaluation on TEST set")
    evaluate_model(model, y_test, y_test_predicted)        
    plot_dataset(X_test, "f0", "f1", y_test) #, "o", title1)
    _ = plt.xlim(xlim), plt.ylim(ylim)
    plot_boundary(model) 
    plot_ellipses(model)       