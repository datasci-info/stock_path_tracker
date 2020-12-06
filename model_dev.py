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

# %%
# %%time
N=5
DIRECTION='is_turnpt'
WEIGHTS_QUNATILE = 0.01

model = get_model(N, DIRECTION, WEIGHTS_QUNATILE, DEPTH=10, LEARNING_RATE=1, L2_LEAF_REG=10*5, OD_WAIT=50, CACHE=True, ADD_DIFF=False)
# path = get_model_path(N, DIRECTION)
# model = load_model(path)

# %%
# %%time
train_pool, test_pool =  get_train_test_pools(N,DIRECTION, WEIGHTS_QUNATILE, CACHE=True)
# best_iter = next(map(lambda x: len(x[1])-1, model.eval_metrics(train_pool,['Precision']).items()))
# model.plot_tree(best_iter)

# %%
print('training pool')
get_precision_recall(model, train_pool)

print('testing pool')
get_precision_recall(model, test_pool)

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
