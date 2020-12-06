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


import numpy as np
from catboost import CatBoostClassifier, Pool
from catboost import CatBoostRegressor

import pickle, io

def get_model(N=1,DIRECTION='is_turnpt',PCT_TRAIN=0.75,N_ITERATIONS=10**5,DEPTH=10,LEARNING_RATE=0.5,LOSS_FUNCTION='RMSE',L2_LEAF_REG=202,OD_TYPE='Iter',OD_WAIT=250):
  df_features = get_featrues(specs)
  df_features.index = df_features.index.date
  df_features = df_features.shift(1)
  # df_features.head()
  # df_features.shape

  df_y = get_df_turnpt_measures(N).fillna(0) # DONE
  df_y.index = df_y.tx_datetime

  df_chosen_y = df_y[[DIRECTION]]#DONE

  # df_y.columns
  # df_chosen_y.head()

  assert type(df_chosen_y.index.values[0]) == type(df_features.index.values[0])

  ### DATA ###
  df = df_chosen_y.join(df_features).dropna()
  w = (
    df[[DIRECTION]].join(df_y.prc_diff).
    assign(weight = lambda x: 1+np.where(x.is_turnpt==0, 0, x.prc_diff.abs())).
    weight
  )

  #df.head()
  #df.shape

  ### Ref: Catboost ###
  # # https://github.com/catboost/tutorials

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

  model = CatBoostClassifier(iterations=N_ITERATIONS, # set very large number and set early stops
                            depth=DEPTH, #fine-tune
                            learning_rate=LEARNING_RATE, #max: 0.5
                            loss_function=LOSS_FUNCTION, 
                            l2_leaf_reg = L2_LEAF_REG, #fine-tune
                            od_type = OD_TYPE,
                            od_wait = OD_WAIT
                          ) #DONE

  model.fit(train_pool, eval_set=test_pool)
  best_iter = model.best_iteration_

  model.plot_tree(best_iter)
  m = model.eval_metrics(test_pool, ['Precision', 'Recall'])

  str_generator =(str(i) for i in [DIRECTION, N,PCT_TRAIN,N_ITERATIONS,DEPTH,LEARNING_RATE,LOSS_FUNCTION,L2_LEAF_REG,OD_TYPE,OD_WAIT,m['Precision'][best_iter],m['Recall'][best_iter]])
  str_result = ", ".join(str_generator)
  print( str_result )
  import pandas as pd

  df_pred = pd.DataFrame({'pred': model.predict(test_pool), 'y': y_test})

  n = params['N']
  DIR = params['DIRECTION']
  with open(f"models/model_{n}_{DIR}.pickle", 'wb+') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
  # df_pred.head()
  # df_pred.pred.hist()
  # df_pred.y.hist()
  return model

def get_param_from_env():
  N = int(os.environ.get('N', '1'))

  DIRECTION = os.environ.get('DIRECTION', 'is_turnpt')

  PCT_TRAIN = eval(os.environ.get('PCT_TRAIN','0.75'))

  N_ITERATIONS = eval(os.environ.get('N_ITER', '10**5'))
  DEPTH = eval(os.environ.get('DEPTH', '10'))
  LEARNING_RATE = eval(os.environ.get('LEARNING_RATE','0.5'))
  LOSS_FUNCTION = os.environ.get('LOSS_FUNCTION','MultiClass')
  L2_LEAF_REG = eval(os.environ.get('L2_LEAF_REG','202'))
  OD_TYPE = os.environ.get('OD_TYPE','Iter')
  OD_WAIT = eval(os.environ.get('OD_WAIT','250'))

  return {'N': N, 'DIRECTION':DIRECTION,'PCT_TRAIN':PCT_TRAIN,'N_ITERATIONS':N_ITERATIONS,'DEPTH':DEPTH,'LEARNING_RATE':LEARNING_RATE,'LOSS_FUNCTION':LOSS_FUNCTION,'L2_LEAF_REG':L2_LEAF_REG,'OD_TYPE':OD_TYPE,'OD_WAIT':OD_WAIT}


if __name__ == "__main__":
  params = get_param_from_env()
  model = get_model(**params)
  
