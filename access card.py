from flask import Flask, render_template
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
#import metrics here
from sklearn.metrics import classification_report

df = pd.read_csv("ID Access Card.csv")
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()
# The Access Cards are heavily skewed we need to solve this issue later.
print('Actives', round(df['ID Activity'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Not Actives', round(df['ID Activity'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
print(df)

df['ID Activity'] = df['ID Activity'].map({'Active': 1, 'Not Active': 0})
print(df)

df.sort_values('ID Activity', inplace=True)
df = df[df['ID Activity'] !=0]
print(df)
print('Actives', round(df['ID Activity'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['Activity'] = rob_scaler.fit_transform(df['ID Activity'].values.reshape(-1,1))
df['Time'] = rob_scaler.fit_transform(df['Day of Month'].values.reshape(-1,1))

df.drop(['Time','Activity'], axis=1, inplace=True)
X = df.drop('ID Activity', axis=1)
y = df['ID Activity']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.
# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the Distribution of the labels


# Turn into an array
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))

app = Flask(__name__)

@app.route('/')
def access_card():

  return render_template('access card.html')



if __name__ == '__main__':
  app.run()