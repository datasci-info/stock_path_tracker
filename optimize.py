import config as cfg
import dolphindb
sess = dolphindb.session()
sess.connect("localhost", 8868, "admin", "123456")
sess.run('''
dailyOptionsBasic = loadTable("dfs://dailyOptions", `dailyOptionsBasic);
txo = select * from dailyOptionsBasic where Contract==`TXO order by Date asc;
''')

import itertools 
import numpy as np
import pandas as pd

def get_df():
    twse = pd.read_msgpack('/data/dataset/twse.msgpack')
    df = twse[cfg.D_START:][['close']].copy()
    df['prc_qtz'] = df.close // cfg.SIZE_CELL * cfg.SIZE_CELL
    df['delta'] = df.prc_qtz.diff()
    settlements = pd.to_datetime(pd.read_csv('settlement_txf.csv').settlement).dt.date.tolist()
    df['maturity'] = df.index.where(df.index.isin(settlements)).to_series().fillna(method='bfill').values
    d2m = df.groupby('maturity').size().sort_index().tolist()
    df['d2m'] = sum([list(reversed(range(d))) for d in d2m], []) + [np.nan] * (df.shape[0] - sum(d2m))
    df = df[cfg.D_START: cfg.D_END]

    df.dropna(inplace=True)
    df.sort_index(ascending=True, inplace=True)

    # probability
    df['delta_cell'] = df.delta // cfg.SIZE_CELL
    df.loc[df.delta_cell >= cfg.MAX_DELTA_CELL, 'delta_cell'] = cfg.MAX_DELTA_CELL
    df.loc[df.delta_cell <= - cfg.MAX_DELTA_CELL , 'delta_cell'] = - cfg.MAX_DELTA_CELL
    return df
gen_sequence = lambda var_tau: itertools.product(range(-cfg.MAX_DELTA_CELL, cfg.MAX_DELTA_CELL+1), repeat=var_tau+1)

def get_realizations(var_lambda, var_tau):
    assert var_tau <= 9, 'the probability matrix is not prepared for var_tau > 9'
    all_feasible_seqs = filter(lambda seq: sum(seq) == var_lambda, gen_sequence(var_tau))
    df_real = pd.DataFrame(list(all_feasible_seqs))
    df_real.index.name = 'realization'
    df_real.columns.name = 'duration'
    return df_real

def get_prob_single_day(df):
    prob_delta_cell = pd.Series(np.nan, index=range(-cfg.MAX_DELTA_CELL, cfg.MAX_DELTA_CELL+1))
    prob_delta_cell.update(df.groupby('delta_cell').size().div(df.shape[0]))
    prob_delta_cell.fillna(0, inplace=True)
    return prob_delta_cell


def get_prob(df_real, prob_delta_cell):
    prob = df_real.replace(prob_delta_cell.to_dict()).product(axis=1).sum()
    return prob


def get_previous_date(date):
    previous_date = sess.run(f'''
    select max(Date) as date 
    from txo
    where Date<date(`{date}) and TradingSession=`Norm;
    ''')
    return previous_date.date.dt.strftime('%Y.%m.%d').values[0]

def get_all_txo_opt_price(date, maturity, right):
    assert date >= '2015.01.05', 'date before 2015.01.05 has no data of normal trading sessions'
    previous_date = get_previous_date(date)
    df_opt = sess.run(f'''
    select Date as date, SettleDate as maturity, StrikePrice as strike, SettlePrice as opt_price
    from txo 
    where TradingSession=`Norm and Right=`{right} and SettleDate=`{maturity} and Date=date(`{previous_date});
    ''')
    return df_opt

def get_all_txo_call_lambda(row):
    date = row.Index.strftime('%Y.%m.%d')
    if date <= '2015.01.05':
        return pd.DataFrame()
    maturity = row.maturity.strftime('%Y%m')
    df_call = get_all_txo_opt_price(date, maturity, right=cfg.RIGHT)
    df_call['date'] = row.Index
    df_call['lamb'] = (df_call.strike - row.prc_qtz).div(cfg.SIZE_CELL)
    df_call['d2m'] = row.d2m
    df_call = df_call[['date', 'maturity', 'lamb', 'opt_price', 'd2m']]
    return df_call

def get_cost(df, cost_cutoff):
    df_call = pd.concat([get_all_txo_call_lambda(row) for row in df.itertuples()])
    df_cost = df_call.groupby(['d2m', 'lamb']).opt_price.apply(lambda x: x.quantile(cfg.CUTOFF_COST)).reset_index()
    df_cost.columns = ['d2m', 'lambd', 'opt_price']
    return df_cost

def cal_profit(lambd):
    return (lambd - 1) * cfg.SIZE_CELL

def cal_cost(var_lambda, var_tau, df_cost):
    c1 = df_cost.query(f'(lambd=={var_lambda-1}) and (d2m=={var_tau})').opt_price.values[0] 
    c2 = df_cost.query(f'(lambd=={var_lambda}) and (d2m=={var_tau})').opt_price.values[0]
    cost = c1 - c2
    print(c1, c2)
    return cost

def optimize(var_lambda, var_tau, df_cost):
    prob = get_realizations(var_lambda, var_tau).pipe(get_prob, prob_delta_cell)
    profit = cal_profit(var_lambda)
    cost = cal_cost(var_lambda, var_tau, df_cost)
    val = prob*profit-cost
    return val

if __name__ == '__main__':
    cost_cutoff = 0.5
    df = get_df()
    prob_delta_cell = get_prob_single_day(df)
    df_cost = get_cost(df, cost_cutoff)

    var_lambda, var_tau = 2, 8
    val = optimize(var_lambda, var_tau, df_cost)

