# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
from pprint import pprint
from IPython.display import display

# %%
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# %% [markdown]
# # Data

# %%
from data_ML import  get_data

# %%
N = 5
DIRECTION = 'is_turnpt_upward'
CACHE = True
ADD_DIFF = 0
PCT_TRAIN = 0.75
y, X, idx, num_train = get_data(N, DIRECTION, CACHE, ADD_DIFF, PCT_TRAIN)
num_test = len(y) - num_train

# %% [markdown]
# # long-only strategy
# - time point: 
#     - t0 收盤時，得到 s0 訊號
#     - t+1 開盤進場（目前簡化先用 t0 收盤價代替）
# - predict TP_upward

# %% [markdown]
# # Model: Logistics regression as baseline
# - reference: https://towardsdatascience.com/weighted-logistic-regression-for-imbalanced-dataset-9a5cd88e68b

# %%
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score

# %%
scaler = MinMaxScaler()
scaler.fit(X)

# %%
X_scaled = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1-PCT_TRAIN, random_state=0)

# %% [markdown]
# ## search hyperparameters

# %% [raw]
# default Loigistic Regression model without tuning any hyperparameters
# ------------------------------------------------------------------------------------
# Accuracy Score: 0.945031712473573
#
# Confusion Matrix:
#         Negative	Positve
# False	446         5
# True	21	        1
#
# Area Under Curve: 0.5171840354767183
# Recall score: 0.045454545454545456

# %% [markdown]
# ## best model

# %%
# %%time
w = {0: 0.03, 1: 1}
model = LogisticRegression(random_state=13, max_iter=10**3, class_weight=w)
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

# performance
print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
print(f'Confusion Matrix:')
display(pd.DataFrame(confusion_matrix(y_test, y_pred), index=['False', 'True'], columns=['Negative', 'Positve']))
print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
print(f'Recall score: {recall_score(y_test,y_pred)}')

# %%
accurency_train = model.score(X_train, y_train, sample_weight=y_train*100+1)
accurency_test = model.score(X_test, y_test, sample_weight=y_test*100+1)
assert (accurency_test - accurency_train) < 0.1, 'accurency differene between training and testing is larger than 10% '

# %%
if __name__ == '__main__':
    from sklearn.model_selection import GridSearchCV

    w = [
        {0: i/1000, 1: 1}
        for i in range(25, 35)
    ]
    crange = np.arange(0.5, 20.0, 0.5)
    hyperparam_grid = {
        "class_weight": w , 
    #     "penalty": ["l1", "l2"],
        "C": crange,
        "fit_intercept": [True, False],
    }
    pprint(hyperparam_grid)

    lg3 = LogisticRegression(random_state=13, max_iter=10**3)
    # define evaluation procedure
    grid = GridSearchCV(lg3,hyperparam_grid, scoring="recall", cv=100, refit=True)
    grid.fit(X_scaled, y)

    print(f'Best score: {grid.best_score_} with param: {grid.best_params_}')

# %%
