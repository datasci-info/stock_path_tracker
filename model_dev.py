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

# %%
# %autoreload 2

# %%
# %matplotlib inline

# %%
import matplotlib.pylab as plt
import pandas as pd

# %%
from model import get_model, get_train_test_pools, get_precision_recall, load_model, get_model_path
from catboost import CatBoostRegressor

# %%
# path = get_model_path(N, DIRECTION)
# model = load_model(path)

# %%
import os
from prepare_X import get_featrues, specs
from turnpt_analysis import get_df_turnpt_measures


import pickle, io
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from catboost import CatBoostRegressor


# %%
# X
CACHE=True
ADD_DIFF = 0
PCT_TRAIN = 0.75
df_features = get_featrues(specs, cache=CACHE, add_diff=ADD_DIFF)
df_features.index = df_features.index.date
df_features = df_features.shift(1)

# y
df_y = get_df_turnpt_measures(N).fillna(0)
df_y.index = df_y.tx_datetime

df_chosen_y = df_y[[DIRECTION]] # before the close of t, the agent will decide whether to enter the market by predicting if t close is a turning point 
assert type(df_chosen_y.index.values[0]) == type(df_features.index.values[0])

df_chosen_y[DIRECTION] = np.where(df_chosen_y[DIRECTION]==-1, 2, df_chosen_y[DIRECTION])

### DATA ###
df = df_chosen_y.join(df_features).dropna()
w = (
df[[DIRECTION]].join(df_y.prc_diff).
assign(weight = lambda x: np.where(x.is_turnpt!=0, x.prc_diff.abs(), np.nan)).
assign(weight = lambda x: 1+x.weight.fillna(0)).
#     assign(weight = lambda x: x.weight.fillna(x.weight.quantile(WEIGHTS_QUNATILE))).
weight
)
num_train = int(df.shape[0] * PCT_TRAIN)# DONE

y = df[DIRECTION] # DONE
y_train = y[: num_train]
y_test = y[num_train:]

w_train = w[: num_train]
w_test = w[num_train:]

X = df.iloc[:, 1:]
X_train = X.iloc[:num_train, :]
X_test = X.iloc[num_train:, :]

train_pool = Pool(X_train, y_train, weight=w_train)
test_pool = Pool(X_test, y_test, weight=w_test)

# %%
# %%time
# train_pool, test_pool =  get_train_test_pools(N,DIRECTION, WEIGHTS_QUNATILE, CACHE=True)
# best_iter = next(map(lambda x: len(x[1])-1, model.eval_metrics(train_pool,['Precision']).items()))
# model.plot_tree(best_iter)

# %%
# %%time
N=5
DIRECTION='is_turnpt'
WEIGHTS_QUNATILE = 0.01

model = CatBoostClassifier(iterations=10**5, # set very large number and set early stops
                          depth=10, #fine-tune
                          learning_rate=0.1, #max: 0.5
                          loss_function='MultiClass', 
                          l2_leaf_reg = 202, #fine-tune
                          od_type = 'Iter',
                          od_wait = 250,
#                           subsample = 0.3,
                          rsm = 0.95,
                           random_strength = 0.95,
                           bagging_temperature = 19.95
                          )

model.fit(train_pool, eval_set=test_pool)

# %%
# print('training pool')
# get_precision_recall(model, train_pool)

# print('testing pool')
# get_precision_recall(model, test_pool)

# %%
df_pred = (
    pd.DataFrame(model.predict(test_pool), columns=['pred']).
    assign(y=test_pool.get_label())
)

df_pred_summary = df_pred.groupby(['y', 'pred']).size().to_frame('occurence').reset_index()
df_pred_summary


df_pred[['pred', 'y']].hist()

# %% [markdown]
# # plot entry and exit

# %%
# def plot_turnpt(price, turnpts, dates_index):
#     index_upward = np.where(turnpts==1)[0]
#     index_downward = np.where(turnpts==-1)[0]
    
#     plt.figure(figsize=(20, 5))
#     plt.plot(dates_index, price, color='black', markevery=index_upward.tolist(), marker='^', markerfacecolor='red', markeredgewidth=0.0)
#     plt.plot(dates_index, price, color='black', markevery=index_downward.tolist(), marker='v', markerfacecolor='green', markeredgewidth=0.0)


# test_df_y = df_y[df_y.index.isin(y_test.index)]
# test_price = test_df_y.close.values
# test_turnpts_true = test_df_y.is_turnpt.values
# test_dates_index = test_df_y.index
# plot_turnpt(test_price, test_turnpts_true, test_dates_index)

# %%
# def get_predict_class(model, pool, threshold = 0):
#     max_prob = model.predict_proba(pool).max(axis=1)
#     turnpts_pred = model.predict(pool).flatten()
#     turnpts_pred_filtered = np.where(max_prob>=threshold, turnpts_pred, 0)
#     return turnpts_pred_filtered

# test_turnpts_pred = get_predict_class(model, test_pool)
# plot_turnpt(test_price, test_turnpts_pred, test_dates_index)

# %%
# # only plot 1 or -1 when prob > thereshold
# # noises during 2018-M7 disappear
# test_turnpts_pred = get_predict_class(model, test_pool, threshold=0.6)
# plot_turnpt(test_price, test_turnpts_pred, test_dates_index)

# %% [markdown]
# # plot NAV

# %%
# test_df_y['is_turnpt_pred'] = get_predict_class(model, test_pool)
# test_df_y.head()

# %%
# # test_df_y
# (
#     test_df_y.query('is_turnpt_pred != 0').loc[:, ['tx_datetime', 'close']].
#     assign(entry = lambda x: x.tx_datetime, 
#            exit = lambda x: x.tx_datetime.shift(-1)).dropna().
#     assign(p_entry = lambda x: x.close.loc[x.entry],# 
# #            p_exit = lambda x: x.close[x.exit],# 
# #            ret = lambda x: x.p_exit.div(x.p_entry)-1
#     )
# #     loc[:, ['entry', 'exit']].
# #     join(test_df_y, how='right')
# )

# %% [markdown]
# # check X

# %% [markdown]
# # show metrics

# %%
