
# coding: utf-8

# In[7]:

import numpy as np
import pandas as pd
import pickle
import sys

import sklearn as sl
from sklearn.externals import joblib


# In[8]:

def main():
    lasso = joblib.load('model.pkl') 
    test = pd.read_csv(sys.argv[1],index_col=False)

    test["jump"] = np.log(test["jump"])
    numerical_data = ['age', 'height', 'weight', 'sleep']
    test[numerical_data] = np.log1p(test[numerical_data])
    test = pd.get_dummies(test, columns=['exercise', 'competitive', 'gender', 'injured', 'color', 'race'])

    X_test = test.drop('jump', axis=1)
    y = test.jump

    lasso.predict(X_test)
    print('Prediction Accuracy:')
    print(lasso.score(X_test,y))

    res = lasso.predict(X_test)
    print('Predicted Results:')
    print(np.exp(res))


# In[ ]:

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Please enter a file name for the test case')
        print('Usage: python lasso_predictor test.csv')
    else:
        main()

