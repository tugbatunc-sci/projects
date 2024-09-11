import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("/Users/tugbatunc/Documents/ITU2023-24 FALL TERM/FIZ437E Stat.Lear.from Data-App.inPhy./pulsar_stars.csv")

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((np.array(x1)-np.array(x2))**2))
    return distance

def minkowski_distance(x1, x2, p=2):
    distance = (np.sum((np.array(x1)-np.array(x2))**p))**(1/p)
    return distance
class KNeighborsClassifier:
    # initiate
    def __init__(self, k=5):
        self.k = k

    # fit function
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # predict function
    def predict(self, x):
        labels = []
        for value in x:
            distances = [euclidean_distance(value, X_train) for X_train in self.X_train]
            k_inds = np.argsort(distances)[:self.k]
            print("k indices:", k_inds)
            classes_k = [self.y_train[i] for i in k_inds]
            mode = stats.mode(classes_k)
            labels.append(mode[0])
        return labels
    
data.info()

data = data.rename(columns={'pmean':"mean_integrated_profile",
       'pstd':"std_deviation_integrated_profile",
       'pkurt':"kurtosis_integrated_profile",
       'pskew':"skewness_integrated_profile", 
        'dmean':"mean_dm_snr_curve",
       'dstd':"std_deviation_dm_snr_curve",
       'dkur':"kurtosis_dm_snr_curve",
       'dskew':"skewness_dm_snr_curve",
       })

data.head()

f,ax=plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,linecolor="blue",fmt=".2f",ax=ax)
plt.show()

g = sns.pairplot(data, hue="ispulsar",palette="husl",diag_kind = "kde",kind = "scatter")

y = data["ispulsar"].values
x_data = data.drop(["ispulsar"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

y = data["ispulsar"].values
x_data = data.drop(["ispulsar"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=500/17898, random_state=42)

# Ensuring the desired shapes
# X_train = X_train[:5000]
# X_test = X_test[:500]

knn = KNeighborsClassifier(n_neighbors =13) # n_neighbors = k
knn.fit(X_train,y_train)
knn_prediction = knn.predict(X_test)
k_values=[1,2,3,5,10,15,20]
score_list = []
for k in k_values:
    knn2 = KNeighborsClassifier(n_neighbors=k)
    knn2.fit(X_train, y_train)
    score_list.append(knn2.score(X_test, y_test))

plt.plot(k_values, score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()

def custom_mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Custom Confusion Matrix calculation
def custom_confusion_matrix(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[true_negatives, false_positives],
                     [false_negatives, true_positives]])

# Custom Classification Report calculation
def custom_classification_report(y_true, y_pred):
    # Calculate metrics - accuracy, precision, recall, f1-score
    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}"
mse_knn = custom_mean_squared_error(y_test, knn_prediction)
cm_knn = custom_confusion_matrix(y_test, knn_prediction)
cr_knn = custom_classification_report(y_test, knn_prediction)

print(mse_knn)
print(cm_knn)
print(cr_knn)

from sklearn.metrics import cohen_kappa_score
cks_knn= cohen_kappa_score(y_test, knn_prediction)

score_and_mse = {'model': "knn classification","Score":knn.score(X_test,y_test),"Cohen Kappa Score":cks_knn,"MSE":mse_knn}

print('Classification report for KNN Classification: \n',cr_knn)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_knn, annot=True, fmt=".1f", cmap="flag", cbar=False)
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.title("KNN Classification Confusion Matrix")
plt.show()

score_and_mse