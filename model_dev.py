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

# %%
from prepare_X import get_featrues, specs

df_features = get_featrues(specs)
df_features.index = df_features.index.date

df_features.head()

df_features.shape

# %%
from turnpt_analysis import get_df_turnpt_measures

N = 1
y = 'is_turnpt'

df_y = get_df_turnpt_measures(N).fillna(0)#NOTE: weighted by abs(prc_diff)
df_y.index = df_y.tx_datetime

df_chosen_y = df_y[[y]]

df_chosen_y.head()

# %% [markdown]
# # data

# %%
assert type(df_chosen_y.index.values[0]) == type(df_features.index.values[0])
df = df_chosen_y.join(df_features).dropna()

# %%
df.head();

# %%
df.shape

# %% [markdown]
# # https://github.com/catboost/tutorials

# %%
import numpy as np
from catboost import CatBoostClassifier, Pool

# %%
from catboost import CatBoostRegressor

# %%
num_train = int(df.shape[0] * 0.75)

# %%
y = df[y] 
y_train = y[: num_train]
y_test = y[num_train:]

# %%
X = df.iloc[:, 1:]
X_train = X.iloc[:num_train, :]
X_test = X.iloc[num_train:, :]

# %%
train_pool = Pool(X_train, y_train, weight=y_train.abs()+1)
test_pool = Pool(X_test, y_test, weight=y_test.abs()+1)

# %%
model = CatBoostClassifier(iterations=10**5, # set very large number and set early stops
                          depth=5, #fine-tune
                          learning_rate=0.5, #max: 0.5
                          loss_function='MultiClass', 
                          l2_leaf_reg = 200, #fine-tune
                          od_type = 'Iter',
                          od_wait = 250
                         )

# %%
model.fit(train_pool, eval_set=test_pool)

# %%
best_iter = model.best_iteration_
print(best_iter)

# %%
model.plot_tree(best_iter)

# %%
m = model.eval_metrics(train_pool, ['Precision', 'Recall'])
# m['Precision'][best_iter], m['Recall'][best_iter]

# %%
preds_class = model.predict(test_pool)
preds_proba = model.predict_proba(test_pool)
preds_raw = model.predict(test_pool, prediction_type='RawFormulaVal')


# %%
def get_precision_recall(model, pool):
    m = model.eval_metrics(pool, ['Precision', 'Recall'])

    df_prec_recall = pd.DataFrame([(class_, f'{m[f"Precision:class={idx}"][best_iter]*100:.2f} %', f'{m[f"Recall:class={idx}"][best_iter]*100:.2f} %') 
                  for idx, class_ in enumerate(model.classes_)
                 ], columns = ['class_', 'Precision', 'Recall'])
    return df_prec_recall


print('training pool')
get_precision_recall(model, train_pool)

print('testing pool')
get_precision_recall(model, test_pool)

# %%
import pandas as pd

# %%
df_pred = (
    pd.DataFrame(model.predict(test_pool), columns=['pred'], index=y_test.index).
    assign(y=y_test)
)

df_pred.pred.hist()
plt.title('pred y')
plt.show()

df_pred.y.hist()
plt.title('true y')
plt.show()

df_pred_summary = df_pred.groupby(['y', 'pred']).size().to_frame('occurence').reset_index()
df_pred_summary

# %%
precision =  df_pred_summary.query('(y==1) and (pred==1)').occurence.iloc[0] / df_pred_summary.query('(y==1)').occurence.sum()
precision

# %%
recall = df_pred_summary.query('(y==1) and (pred==1)').occurence.iloc[0] / df_pred_summary.query('(y==1)').occurence.sum() 
recall


# %% [markdown]
# # plot entry and exit

# %%
def plot_turnpt(price, turnpts, dates_index):
    index_upward = np.where(turnpts==1)[0]
    index_downward = np.where(turnpts==-1)[0]
    
    plt.figure(figsize=(20, 5))
    plt.plot(dates_index, price, color='black', markevery=index_upward.tolist(), marker='^', markerfacecolor='red', markeredgewidth=0.0)
    plt.plot(dates_index, price, color='black', markevery=index_downward.tolist(), marker='v', markerfacecolor='green', markeredgewidth=0.0)


test_df_y = df_y[df_y.index.isin(y_test.index)]
test_price = test_df_y.close.values
test_turnpts_true = test_df_y.is_turnpt.values
test_dates_index = test_df_y.index
plot_turnpt(test_price, test_turnpts_true, test_dates_index)


# %%
def get_predict_class(model, pool, threshold = 0):
    max_prob = model.predict_proba(pool).max(axis=1)
    turnpts_pred = model.predict(pool).flatten()
    turnpts_pred_filtered = np.where(max_prob>=threshold, turnpts_pred, 0)
    return turnpts_pred_filtered

test_turnpts_pred = get_predict_class(model, test_pool)
plot_turnpt(test_price, test_turnpts_pred, test_dates_index)

# %%
# only plot 1 or -1 when prob > thereshold
# noises during 2018-M7 disappear
test_turnpts_pred = get_predict_class(model, test_pool, threshold=0.6)
plot_turnpt(test_price, test_turnpts_pred, test_dates_index)

# %% [markdown]
# # plot NAV

# %%
test_df_y['is_turnpt_pred'] = get_predict_class(model, test_pool)
test_df_y.head()

# %%
# test_df_y
(
    test_df_y.query('is_turnpt_pred != 0').loc[:, ['tx_datetime', 'close']].
    assign(entry = lambda x: x.tx_datetime, 
           exit = lambda x: x.tx_datetime.shift(-1)).dropna().
    assign(p_entry = lambda x: x.close.loc[x.entry],# 
#            p_exit = lambda x: x.close[x.exit],# 
#            ret = lambda x: x.p_exit.div(x.p_entry)-1
    )
#     loc[:, ['entry', 'exit']].
#     join(test_df_y, how='right')
)

# %%
signal = pd.Series(1, index=ret.index)
signal

# %%
# test_df_y
df_data = test_df_y.copy()

ret = df_data.close.pct_change().fillna(0)
nav = (1 + ret * signal).cumprod().plot(title='NAV')


# %%
signal.

# %%

# %% [markdown]
# # show metrics

# %%
