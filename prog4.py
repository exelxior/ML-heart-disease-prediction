""" CS 461 - BINH NGUYEN - 16168354 - PROF. BRIAN HARE """
# data processing, CSV file I/O
import pandas as pd
# linear algebra
import numpy as np
# Deep Learning Libraries
import tensorflow as tf
# Misc. Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
# Turn off TF warnings
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load "heart.csv" data
heart_data_df = pd.read_csv("heart.csv")

# Splitting data into features (X) and label/target (Y)
X = heart_data_df.drop("target", axis=1)
Y = heart_data_df["target"]

# Split data set into 15% test and 85% training (which will be splitted during training)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)

# One-hot encoding for categorical data
X = pd.get_dummies(X, prefix_sep='_', drop_first=False,
                   columns=["sex", "cp", "fbs", "restecg", "exang", "slope", "ca"])
X = pd.get_dummies(X, prefix_sep='_', drop_first=True,
                   columns=['thal']) # thal is 1-3 so drop_first=True

# Standardising/Normalizing true number data
continuous_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
sc = StandardScaler(copy=False)
X[continuous_columns] = sc.fit_transform(X[continuous_columns])

# Create and compile our model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=15, activation="relu"),
        tf.keras.layers.Dense(units=1, activation="softmax"),
    ]
)
model.compile(optimizer="adam", loss='binary_crossentropy' , metrics=["acc"])

# load training data and split into 7 folds (close to 15% validation set) for cross validation
folds = list(StratifiedKFold(n_splits=7, shuffle=True, random_state=1).split(X_train, y_train))

# train our model using K-fold cross validation
for j, (train_idx, val_idx) in enumerate(folds):
    print('\nFold ', j)
    X_train_cv = X_train.iloc[train_idx]
    y_train_cv = y_train.iloc[train_idx]
    X_valid_cv = X_train.iloc[val_idx]
    y_valid_cv = y_train.iloc[val_idx]

    model.fit(np.array(X_train_cv), np.array(y_train_cv), epochs=200,
              validation_data=(np.array(X_valid_cv), np.array(y_valid_cv)))
    print(model.evaluate(np.array(X_valid_cv), np.array(y_valid_cv)))

# evaluate against test data
model.evaluate(np.array(X_test), np.array(y_test))

# Let's try other configurations of NN
# RELU & SIGMOID
model2 = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=15 , activation="relu"),
        tf.keras.layers.Dense(units=1 , activation="sigmoid"),
    ]
)
model2.compile(optimizer="adam", loss='binary_crossentropy', metrics=["acc"])
model2.fit(np.array(X_train), np.array(y_train) , epochs=200, validation_split=0.2) # 0.2 validation split is close to 15% of the entire data set
model2.evaluate(np.array(X_test), np.array(y_test))

# TANH & SIGMOID
model3 = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=15, activation="tanh"),
        tf.keras.layers.Dense(units=1, activation="sigmoid"),
    ]
)
model3.compile(optimizer="adam", loss='binary_crossentropy', metrics=["acc"])
model3.fit(np.array(X_train), np.array(y_train), epochs=200, validation_split=0.2) # 0.2 validation split is close to 15% of the entire data set
model3.evaluate(np.array(X_test), np.array(y_test))

# RELU & SIGMOID with more units
model4 = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=50, activation="relu"),
        tf.keras.layers.Dense(units=1, activation="sigmoid"),
    ]
)
model4.compile(optimizer="adam", loss='binary_crossentropy', metrics=["acc"])
model4.fit(np.array(X_train), np.array(y_train), epochs=200, validation_split=0.2) # 0.2 validation split is close to 15% of the entire data set
model4.evaluate(np.array(X_test), np.array(y_test))

# RELU & SOFTMAX no K-fold CV
model5 = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units=15 , activation="relu"),
        tf.keras.layers.Dense(units=1 , activation="softmax"),
    ]
)
model5.compile(optimizer="adam", loss='binary_crossentropy', metrics=["acc"])
model5.fit(np.array(X_train), np.array(y_train) , epochs=200, validation_split=0.2) # 0.2 validation split is close to 15% of the entire data set
model5.evaluate(np.array(X_test), np.array(y_test))
