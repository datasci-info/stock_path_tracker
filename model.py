# %%
# %load_ext autoreload

# %%
# %autoreload 2

# %%
# %matplotlib inline

# %%
import os
from prepare_X import get_featrues, specs
from turnpt_analysis import get_df_turnpt_measures


import pickle, io
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from catboost import CatBoostRegressor



def get_train_test_pools_with_index(N=1,DIRECTION='is_turnpt',PCT_TRAIN=0.75, WEIGHTS_QUNATILE=0.01, CACHE=False, ADD_DIFF=False):
  # X
  df_features = get_featrues(specs, cache=CACHE, add_diff=ADD_DIFF)
  df_features.index = df_features.index.date
  df_features = df_features.shift(1)
  
  # y
  df_y = get_df_turnpt_measures(N).fillna(0)
  df_y.index = df_y.tx_datetime

  df_chosen_y = df_y[[DIRECTION]] # before the close of t, the agent will decide whether to enter the market by predicting if t close is a turning point 
  assert type(df_chosen_y.index.values[0]) == type(df_features.index.values[0])

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
  train_index, test_index = y_train.index, y_test.index

  w_train = w[: num_train]
  w_test = w[num_train:]

  X = df.iloc[:, 1:]
  X_train = X.iloc[:num_train, :]
  X_test = X.iloc[num_train:, :]

  train_pool = Pool(X_train, y_train, weight=w_train)
  test_pool = Pool(X_test, y_test, weight=w_test)
  return train_pool, test_pool, train_index, test_index

def get_train_test_pools(N=1,DIRECTION='is_turnpt',PCT_TRAIN=0.75, WEIGHTS_QUNATILE=0.01, CACHE=False, ADD_DIFF=False):
  # X
  df_features = get_featrues(specs, cache=CACHE, add_diff=ADD_DIFF)
  df_features.index = df_features.index.date
  df_features = df_features.shift(1)
  
  # y
  df_y = get_df_turnpt_measures(N).fillna(0)
  df_y.index = df_y.tx_datetime

  df_chosen_y = df_y[[DIRECTION]] # before the close of t, the agent will decide whether to enter the market by predicting if t close is a turning point 
  assert type(df_chosen_y.index.values[0]) == type(df_features.index.values[0])

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
  return train_pool, test_pool
    
def get_model(N=1,DIRECTION='is_turnpt',PCT_TRAIN=0.75,WEIGHTS_QUNATILE=0.1,CACHE=False,ADD_DIFF=False,N_ITERATIONS=10**5,DEPTH=10,LEARNING_RATE=0.5,LOSS_FUNCTION='MultiClass',L2_LEAF_REG=202,OD_TYPE='Iter',OD_WAIT=250):
 
  train_pool, test_pool =  get_train_test_pools(N, DIRECTION, PCT_TRAIN, WEIGHTS_QUNATILE, CACHE, ADD_DIFF)
  model = CatBoostClassifier(iterations=N_ITERATIONS, # set very large number and set early stops
                            depth=DEPTH, #fine-tune
                            learning_rate=LEARNING_RATE, #max: 0.5
                            loss_function=LOSS_FUNCTION, 
                            l2_leaf_reg = L2_LEAF_REG, #fine-tune
                            od_type = OD_TYPE,
                            od_wait = OD_WAIT
                          ) #DONE

  model.fit(train_pool, eval_set=test_pool)
  model.save_model(f"models/model_{N}_{DIRECTION}.sav")
  return model

def get_precision_recall(model, pool):
    m = model.eval_metrics(pool, ['Precision', 'Recall'])

    df_prec_recall = pd.DataFrame([(class_, f'{m[f"Precision:class={idx}"][-1]*100:.2f} %', f'{m[f"Recall:class={idx}"][-1]*100:.2f} %') 
                  for idx, class_ in enumerate(model.classes_)
                 ], columns = ['class_', 'Precision', 'Recall'])
    return df_prec_recall

def get_param_from_env():
  N = int(os.environ.get('N', '5'))

  DIRECTION = os.environ.get('DIRECTION', 'is_turnpt')

  PCT_TRAIN = eval(os.environ.get('PCT_TRAIN','0.75'))
  WEIGHTS_QUNATILE = eval(os.environ.get('WEIGHTS_QUNATILE','0.01'))

  N_ITERATIONS = eval(os.environ.get('N_ITER', '10**5'))
  DEPTH = eval(os.environ.get('DEPTH', '10'))
  LEARNING_RATE = eval(os.environ.get('LEARNING_RATE','0.5'))
  LOSS_FUNCTION = os.environ.get('LOSS_FUNCTION','MultiClass')
  L2_LEAF_REG = eval(os.environ.get('L2_LEAF_REG','202'))
  OD_TYPE = os.environ.get('OD_TYPE','Iter')
  OD_WAIT = eval(os.environ.get('OD_WAIT','250'))

  return {'N': N, 'DIRECTION':DIRECTION,'PCT_TRAIN':PCT_TRAIN,'N_ITERATIONS':N_ITERATIONS,'DEPTH':DEPTH,'LEARNING_RATE':LEARNING_RATE,'LOSS_FUNCTION':LOSS_FUNCTION,'L2_LEAF_REG':L2_LEAF_REG,'OD_TYPE':OD_TYPE,'OD_WAIT':OD_WAIT}

def load_model(path):
    model = CatBoostClassifier()
    model.load_model(path)
    return model

def get_model_path(N, DIRECTION):
    path = f"models/model_{N}_{DIRECTION}.sav"
    return path

def get_best_iteration(model, N, DIRECTION, CACHE=False):
    train_pool, test_pool =  get_train_test_pools(N,DIRECTION, WEIGHTS_QUNATILE, CACHE)
    best_iter = next(map(lambda x: len(x[1])-1, model.eval_metrics(train_pool,['Precision']).items()))
    return best_iter


if __name__ == "__main__":
  params = get_param_from_env()
  model = get_model(**params)
  
