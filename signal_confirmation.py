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

# %%
from model_logistic_regression import y, X, idx, num_test
from model_logistic_regression import scaler, model
from turnpt_analysis import get_daily_TXF_OHLC

# %%
X_scaled = scaler.transform(X)
X_, y_, idx_ = X_scaled[-num_test:], y[-num_test:], idx[-num_test:]

# %%
signal = model.predict(X_)
signal


# %%
def get_position(df, holding_days=5):
    df['exit'] = 0
    df['exit_at'] = np.nan
    df['exit_at'] = np.where((df.enter == 1) & (df.p0 > df.p5), 'p5', df.exit_at)# NOTE: order is important
    df['exit_at'] = np.where((df.enter == 1) & (df.p0 > df.p4), 'p4', df.exit_at)
    df['exit_at'] = np.where((df.enter == 1) & (df.p0 > df.p3), 'p3', df.exit_at)
    df['exit_at'] = np.where((df.enter == 1) & (df.p0 > df.p2), 'p2', df.exit_at)
    df['exit_at'] = np.where((df.enter == 1) & (df.p0 > df.p1), 'p1', 'p_holding')
    assert df.exit_at.isna().sum() == 0
    
    df['prc_exit'] = np.nan
    df['prc_exit'] = np.where(df.exit_at == 'p5', df.p5, df.prc_exit)
    df['prc_exit'] = np.where(df.exit_at == 'p4', df.p4, df.prc_exit)
    df['prc_exit'] = np.where(df.exit_at == 'p3', df.p3, df.prc_exit)
    df['prc_exit'] = np.where(df.exit_at == 'p2', df.p2, df.prc_exit)
    df['prc_exit'] = np.where(df.exit_at == 'p1', df.p1, df.prc_exit)
    df['prc_exit'] = np.where(df.exit_at == 'p_holding', df.p0.shift(-holding_days), df.prc_exit)
    
    days_enter = df.query('enter == 1').index
    nrows = df.shape[0]
    for idx_row in range(nrows):
        dt, row = df.index[idx_row], df.iloc[idx_row]
        if dt in days_enter:
            if row.exit_at == 'p_holding':
                idx_exit = idx_row+holding_days
            elif row.exit_at == 'p1':
                idx_exit = idx_row+1
            elif row.exit_at == 'p2':
                idx_exit = idx_row+2
            elif row.exit_at == 'p3':
                idx_exit = idx_row+3
            elif row.exit_at == 'p4':
                idx_exit = idx_row+4
                
            elif row.exit_at == 'p5':
                idx_exit = idx_row+5
    
            else:
                raise ValueError(f'There is no such exit_at code: {row.exit_at}')
                
            
            if idx_exit <= nrows-1:
                dt_exit = df.index[idx_exit]
                df.loc[dt_exit, 'exit'] += 1

    buy = df.enter.cumsum()
    sell = df.exit.cumsum()
    df['position'] = buy - sell
    return df


def get_df_action(dt_includes, signal, holding_days):
    df = (
        get_daily_TXF_OHLC().
        set_index('tx_datetime').loc[:, ['close']].
        query('index in @idx_').
        assign(signal = signal, 
               signal_true = y_).
        assign(p0 = lambda x: x.close,
               p_1 = lambda x: x.close.shift(1), p_2 = lambda x: x.close.shift(2), p_3 = lambda x: x.close.shift(3), p_4 = lambda x: x.close.shift(4), p_5 = lambda x: x.close.shift(5),
               p1 = lambda x: x.close.shift(-1), p2 = lambda x: x.close.shift(-2), p3 = lambda x: x.close.shift(-3), p4 = lambda x: x.close.shift(-4), p5 = lambda x: x.close.shift(-5),).
        dropna().
        assign(enter = lambda x: np.where(x[['p0', 'p_1', 'p_2', 'p_3', 'p_4', 'p_5']].idxmin(axis=1) == 'p0', x.signal, 0)).
        pipe(get_position, holding_days = holding_days).
        assign(action = lambda x: x.position.diff().fillna(x.iloc[0].position))
    )
    return df


# %%
if __name__ == '__main__':
    holding_days = 5
    df_action = get_df_action(idx_, signal, holding_days)
    display(df_action.head(15).loc[:, ['signal_true', 'signal', 'enter', 'exit_at', 'position', 'action']])
    
    # correlation
    df_signal_confirmed = (
        df_action.
        assign(signal_enter = lambda x: x.enter == 1,
               signal_t1 = lambda x: ((x.enter == 1) & (~x.exit_at.isin(['p1']))).astype(int),
               signal_t2 = lambda x: ((x.enter == 1) & (~x.exit_at.isin(['p1', 'p2']))).astype(int),
               signal_t3 = lambda x: ((x.enter == 1) & (~x.exit_at.isin(['p1', 'p2', 'p3']))).astype(int),
               signal_t4 = lambda x: ((x.enter == 1) & (~x.exit_at.isin(['p1', 'p2', 'p3', 'p4']))).astype(int),
               signal_t5 = lambda x: ((x.enter == 1) & (~x.exit_at.isin(['p1', 'p2', 'p3', 'p4', 'p5']))).astype(int)).
        loc[:, ['signal_true', 'signal', 'signal_enter', 'signal_t1', 'signal_t2', 'signal_t3', 'signal_t4', 'signal_t5']]
    )
    display(round(df_signal_confirmed.corr(), 2))


# %%

# %%

# %%

# %%
