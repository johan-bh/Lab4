import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer_data = load_breast_cancer()

H = 5

#Prints the dataset's feature names
# print((cancer_data['feature_names']))

device = torch.device('cpu')

#Converts the Scikit-Learn Bunch object to a Pandas DF
cancer_df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
cancer_df['target'] = pd.Series(cancer_data.target)

# print(cancer_df)

#Store data in X,y
X = cancer_df[cancer_df.columns.drop('target')]
y = cancer_df['target']

# print(X)
x = torch.tensor(np.expand_dims(X,1), dtype=torch.float32, device=device)
# print(y)

#Create train/test split
# X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)

#Create KNN-model and set no. of neighbors to 1
KNN = KNeighborsClassifier(n_neighbors=13)

#Train the model on the training data
KNN.fit(X_train, y_train)

#Make predictions on X_test using the trained KNN-model
#print(KNN.predict(X_test))

#Test the KNN-model on the test data and print the test accuracy
print(KNN.score(X_test,y_test))