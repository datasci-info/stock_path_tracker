#!/usr/bin/env python
# coding: utf-8

# In[128]:

from pprint import pprint
from IPython.display import display

import pandas as pd
import numpy as np
import os

from prepare_X import get_featrues, specs
from turnpt_analysis import get_df_turnpt_measures



def get_data(N, DIRECTION, CACHE, ADD_DIFF, PCT_TRAIN):
    # X
    df_features = get_featrues(specs, cache=CACHE, add_diff=ADD_DIFF)
    df_features.index = df_features.index.date

    # y
    df_y = get_df_turnpt_measures(N).fillna(0)
    df_y.index = df_y.tx_datetime

    df_chosen_y = df_y[[DIRECTION]] # before the close of t, the agent will decide whether to enter the market by predicting if t close is a turning point 
    assert type(df_chosen_y.index.values[0]) == type(df_features.index.values[0]), 'both indice should be of datatime type'

    ### DATA ###
    df = df_chosen_y.join(df_features).dropna()

    num_train = int(df.shape[0] * PCT_TRAIN)# DONE
    idx = df.index
    y = df[DIRECTION].values # DONE
    X = df.values[:, 1:]
    return y, X, idx, num_train

if __name__ == '__main__':
    N = int(os.environ.get('N', '5'))
    DIRECTION = os.environ.get('DIRECTION', 'is_turnpt_upward')
    CACHE = bool(int(os.environ.get('CACHE', '1')))
    ADD_DIFF = bool(int(os.environ.get('ADD_DIFF', '0')))
    PCT_TRAIN = float(os.environ.get('PCT_TRAIN', '0.75'))
    y, X, idx, num_train = get_data(N, DIRECTION, CACHE, ADD_DIFF, PCT_TRAIN)

