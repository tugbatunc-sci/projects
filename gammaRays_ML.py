import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import colormaps
from sklearn import datasets


# # Summary of Dataset

# Using imaging techniques, Monte Carlo (MC) simulation is used to artificially generate the data, which replicate the registration of high-energy gamma particles in a ground-based atmospheric Cherenkov gamma telescope. By detecting the Cherenkov radiation released by charged particles formed within electromagnetic showers started by the gamma rays in the atmosphere, this telescope examines high-energy gamma rays. Reconstructing shower parameters is made possible by the detector's recording of the visible to ultraviolet (UV) Cherenkov radiation that enters the atmosphere.
# 
# **Parameters**
# 1. fLength:  continuous  # major axis of ellipse [mm]
# 2.  fWidth:   continuous  # minor axis of ellipse [mm] 
# 3.  fSize:    continuous  # 10-log of sum of content of all pixels [in #phot]
# 4.  fConc:    continuous  # ratio of sum of two highest pixels over fSize  [ratio]
# 5.  fConc1:   continuous  # ratio of highest pixel over fSize  [ratio]
# 6.  fAsym:    continuous  # distance from highest pixel to center, projected onto major axis [mm]
# 7.  fM3Long:  continuous  # 3rd root of third moment along major axis  [mm] 
# 8.  fM3Trans: continuous  # 3rd root of third moment along minor axis  [mm]
# 9.  fAlpha:   continuous  # angle of major axis with vector to origin [deg]
# 10.  fDist:    continuous  # distance from origin to center of ellipse [mm]
# 11.  class:    g,h         # gamma (signal), hadron (background)
# 
# **Counts of each class**
# * g:   12332
# * h:    6688

# # Data Collection and Processing

# Loading the data from csv file to Pandas DataFrame
data_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long','fM3Trans', 'fAlpha', 'fDist', 'class']
telescope_data = pd.read_csv('/Users/tugbatunc/Documents/ITU2023-24 FALL TERM/FIZ437E Stat.Lear.from Data-App.inPhy./PROJECT/magic.data', sep=',', header=None, names=data_names)

# Printing the first 5 rows of the dataframe
telescope_data.head()
# Printing the last 5 rows of the dataframe
telescope_data.tail()
# Understanding shape of data
telescope_data.shape
# Getting some informations about features and labels
telescope_data.info()
# Checking for the missing value in each features
telescope_data.isnull().sum()
# If there would be features with many missing elements, then we would have drop those features to obtain more accurate classification.
# Getting some statistical measures about the data
telescope_data.describe()
# Finding the classicications of gamma as signal (g) and hadron shower as background noise (0).
telescope_data['class'].value_counts()
# 'Class' consists of strings: g and h. To investigate data further and process, I will consider gamma, g (signal) as 1 and hadron shower, h (background) as 0.

class_mapping = {"g": 1, "h": 0}
telescope_data['class'] = telescope_data['class'].map(class_mapping)
telescope_data.head()
telescope_data.tail()

# Defining features X and label y.
X = telescope_data.drop(columns = ['class'], axis=1)
y = telescope_data['class']
# Splitting data as train and test
X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(),
                                                    test_size=0.2, random_state=123)

# Reshaping data (we use wX.T instead of wX.T + b)
X_train = np.c_[np.ones(len(X_train)), X_train]
X_test = np.c_[np.ones(len(X_test)), X_test]
y_test = y_test[:, np.newaxis].T
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Looking closer X values (features or inputs)
telescope_data_features = telescope_data.drop(columns='class')
print(telescope_data_features)
# Compute pairwise correlation of features .
correlation = telescope_data.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Pairwise Correlation Heatmap')
plt.show()
g = sns.pairplot(telescope_data, hue="class",palette="husl",diag_kind = "kde",kind = "scatter")

# Creating a 4x3 grid of subplots
fig, axs = plt.subplots(4, 3, figsize = (20, 30))

# Defining numbers of column and row
row_nums = [row for row in range(0,4)]
col_nums = [col for col in range(0,3)]

# Getting column names from the DataFrame
telescope_data_columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long','fM3Trans', 'fAlpha', 'fDist', 'class']

# Iterating each row and column to create histograms
for row_num, i in zip(row_nums, range(0, 10, 3)):
    for col_num, y in zip(col_nums, range(0, 3)):
        # Check if the index is within the range of columns
        if i + y < 10:
            # Create a histogram using Seaborn
            sns.histplot(data=telescope_data, x=telescope_data_columns[i + y], hue=telescope_data['class'].astype(str)
                         , kde=True, ax=axs[row_num, col_num], bins=10)

# Removing unnecessary subplots in the last row
fig.delaxes(axs[3][1])
fig.delaxes(axs[3][2])

# Setting details
plt.suptitle('Distribution of the Properties Measured for Each Class', fontsize=18)
plt.subplots_adjust(top=0.96)

plt.show()


# Problem statement
# Given the MAGIC gamma telescope that is generated to simulate registration of high energy gamma particles in a ground-based atmospheric Cherenkov gamma telescope using the imaging technique, develop and evaluate different classification models to accurately classify instances into the "gamma" and "hadron" classes using the MAGIC gamma telescope dataset.
# Model Selection
# Plotting confusion matrix
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# Gaussian Naive Bayes
from naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train , y_train)
X_test_prediction = model.predict(X_test)
y_test_reshaped = y_test.reshape(-1, 1)
y_train_reshaped = y_train.reshape(-1, 1)
print("Accuracy score :", accuracy_score(y_test_reshaped, X_test_prediction))
plot_confusion_matrix(y_test_reshaped, X_test_prediction, "Gaussian Naive Bayes Confusion Matrix")

# Support Vector Machine
from svm import SVM
model_2 = SVM()
model_2.fit(X_train , y_train)
X_test_prediction_2 = model_2.predict(X_test)
y_test_reshaped = y_test.reshape(-1, 1)
y_train_reshaped = y_train.reshape(-1, 1)
print("Accuracy score :", accuracy_score(y_test_reshaped, X_test_prediction_2))
plot_confusion_matrix(y_test_reshaped, X_test_prediction_2, "SVM Confusion Matrix")

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
model_3 = DecisionTreeClassifier(random_state=123)
model_3.fit(X_train, y_train)
X_test_prediction_3 = model_3.predict(X_test)
print("Accuracy score :", accuracy_score(y_test_reshaped, X_test_prediction_3))
plot_confusion_matrix(y_test_reshaped, X_test_prediction_3, "Decision Tree Confusion Matrix")

# k-Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
model_4 = KNeighborsClassifier()
model_4.fit(X_train, y_train)
X_test_prediction_4 = model_4.predict(X_test)
print("Accuracy score :", accuracy_score(y_test_reshaped, X_test_prediction_4))
plot_confusion_matrix(y_test_reshaped, X_test_prediction_4, "K-Neighbors Classifier Confusion Matrix")

# Random Forest
from sklearn.ensemble import RandomForestClassifier
model_5 = RandomForestClassifier(random_state=123)
model_5.fit(X_train, y_train)
X_test_prediction_5 = model_5.predict(X_test)
print("Accuracy score :", accuracy_score(y_test_reshaped, X_test_prediction_5))
plot_confusion_matrix(y_test_reshaped, X_test_prediction_5, "Random Forest Classifier Confusion Matrix")

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
model_6 = GradientBoostingClassifier(random_state=123)
model_6.fit(X_train, y_train)
X_test_prediction_6 = model_6.predict(X_test)
print("Accuracy score :", accuracy_score(y_test_reshaped, X_test_prediction_6))
plot_confusion_matrix(y_test_reshaped, X_test_prediction_6, "Gradient Boosting Classifier Confusion Matrix")


# Highest accuracy is obtained with Random Forest Classifier and Gradient Boosting Classifier. Now, I will compare it with neural network by TensorFlow - keras implementation.

# Neural Network
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Loading the data from csv file to Pandas DataFrame
data_names_nn = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long','fM3Trans', 'fAlpha', 'fDist', 'class']
telescope_data_nn = pd.read_csv('/Users/tugbatunc/Documents/ITU2023-24 FALL TERM/FIZ437E Stat.Lear.from Data-App.inPhy./PROJECT/magic.data', sep=',', header=None, names=data_names)

class_mapping_nn = {"g": 1, "h": 0}
telescope_data_nn['class'] = telescope_data_nn['class'].map(class_mapping_nn)

# Defining features X and y
X_nn = telescope_data_nn.drop(columns = ['class'], axis=1)
y_nn = telescope_data_nn['class']

# Convert labels to one-hot encoding
num_classes = len(y_nn.unique())
y_one_hot = to_categorical(y_nn, num_classes)

# Split the data - this time splitting in the format of wx+b
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_nn, y_one_hot, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_nn)
X_test_scaled = scaler.transform(X_test_nn)

# Neural network model
model = Sequential()
model.add(Dense(32, input_shape=(X_train_scaled.shape[1],), activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Output layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(X_train_scaled, y_train_nn, epochs=100, batch_size=5, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test_nn)
print(f"Test Accuracy: {accuracy*100:.2f}%")
