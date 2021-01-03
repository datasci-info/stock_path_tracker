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
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# %%
from signal_confirmation import signal, idx_, get_df_action


# %%
def get_PnL(idx_, signal, holding_days):
    df_action = get_df_action(idx_, signal, holding_days)
    df = (
        df_action.
        assign(prc_diff = lambda x: x.prc_exit - x.p0,
               PnL = lambda x: x.prc_diff.fillna(0).cumsum()).
        loc[:, ['action', 'close', 'prc_diff', 'PnL']]
    )
    df.PnL.plot(title=f'holding days: {holding_days}', figsize=(20, 5))
    plt.show()
    return df


# %%
holding_days = 5
df_PnL = get_PnL(idx_, signal, holding_days)

# %%
holding_days = 10
df_PnL = get_PnL(idx_, signal, holding_days)

# %%
holding_days = 15
df_PnL = get_PnL(idx_, signal, holding_days)

# %%
holding_days = 20
df_PnL = get_PnL(idx_, signal, holding_days)

# %%

# %%

# %%

# %%
