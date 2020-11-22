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
from findataflow.dataprocs import feature_generator as fg
from findataflow.dataprocs import resample as resampler

# %%
import pandas as pd

# %%
specs = []

# %%
# BB
FUNCS = {
    'price': ['BULL_MA', 'BULL_UP', 'BULL_DOWN'],}
PARAMETER = [
    {
        'timeperiod':5,
        'nbdevup': 2, 
        'nbdevdn': 2
    }
]
specs += [
        {
            'symbol': 'TXF',
            'type': func_type,
            'freq': '1D',
            'func': func, 
            'args': param
        } 
        for param in PARAMETER
        for func_type, funcs in FUNCS.items()
        for func in funcs
    ]


# %%
# BIAS
FUNCS = {
    'swig': ['BIAS'],
}
assert all(func_type in ['price', 'swig', 'vol'] for func_type in FUNCS.keys())
PARAMETER = [
    {'timeperiod': 5}, {'timeperiod': 10}, {'timeperiod': 15}, 
         {'timeperiod': 20}, {'timeperiod': 25}, {'timeperiod': 30}
]
specs += [
        {
            'symbol': 'TXF',
            'type': func_type,
            'freq': '1D',
            'func': func, 
            'args': param
        } 
        for param in PARAMETER
        for func_type, funcs in FUNCS.items()
        for func in funcs
    ]


# %%
# candlestick
FUNCS = {
    'price': ['candlestick'],
}
assert all(func_type in ['price', 'swig', 'vol'] for func_type in FUNCS.keys())
PARAMETER = [
    {}
]

specs += [
        {
            'symbol': 'TXF',
            'type': func_type,
            'freq': '1D',
            'func': func, 
            'args': param
        } 
        for param in PARAMETER
        for func_type, funcs in FUNCS.items()
        for func in funcs
    ]


# %%
# CDP
FUNCS = {
    'price': [
        'CDP', 'AH', 'AL', 'NH', 'NL'
    ],
}
assert all(func_type in ['price', 'swig', 'vol'] for func_type in FUNCS.keys())
PARAMETER = [
    {}
]

specs += [
        {
            'symbol': 'TXF',
            'type': func_type,
            'freq': '1D',
            'func': func, 
            'args': param
        } 
        for param in PARAMETER
        for func_type, funcs in FUNCS.items()
        for func in funcs
    ]


# %%
# DMI 
FUNCS = {
    'swig': [
        'plusDM', 'minusDM', 'plusDI', 'minusDI', 'DX', 'ADX'
    ],
}
assert all(func_type in ['price', 'swig', 'vol'] for func_type in FUNCS.keys())
PARAMETER = [
    {'timeperiod': 5}, {'timeperiod': 10}, {'timeperiod': 15}, 
         {'timeperiod': 20}, {'timeperiod': 25}, {'timeperiod': 30}
]
specs += [
        {
            'symbol': 'TXF',
            'type': func_type,
            'freq': '1D',
            'func': func, 
            'args': param
        } 
        for param in PARAMETER
        for func_type, funcs in FUNCS.items()
        for func in funcs
    ]


# %%
# KD
FUNCS = {
    'swig': [
        'RSV', 'K', 'D'
    ],
}
assert all(func_type in ['price', 'swig', 'vol'] for func_type in FUNCS.keys())
PARAMETER = [
    {
        'fastk_period': 9,
        'slowk_period': 3,
        'slowd_period': 3
    },
]
specs += [
        {
            'symbol': 'TXF',
            'type': func_type,
            'freq': '1D',
            'func': func, 
            'args': param
        } 
        for param in PARAMETER
        for func_type, funcs in FUNCS.items()
        for func in funcs
    ]


# %%
# MACD
FUNCS = {
    'swig': [
        'MACD', 'MACD_SIGNAL', 'MACD_HIST'
            ],
}
assert all(func_type in ['price', 'swig', 'vol'] for func_type in FUNCS.keys())
PARAMETER = [
    {
        'fastperiod': 12, 
        'slowperiod': 26, 
        'signalperiod': 9
    }, 
]
specs += [
        {
            'symbol': 'TXF',
            'type': func_type,
            'freq': '1D',
            'func': func, 
            'args': param
        } 
        for param in PARAMETER
        for func_type, funcs in FUNCS.items()
        for func in funcs
    ]


# %%
# MFI
FUNCS = {
    'swig': ['MFI'],
}
assert all(func_type in ['price', 'swig', 'vol'] for func_type in FUNCS.keys())
PARAMETER = [
    {'timeperiod': 5}, {'timeperiod': 10}, {'timeperiod': 15}, 
         {'timeperiod': 20}, {'timeperiod': 25}, {'timeperiod': 30}
]
specs += [
        {
            'symbol': 'TXF',
            'type': func_type,
            'freq': '1D',
            'func': func, 
            'args': param
        } 
        for param in PARAMETER
        for func_type, funcs in FUNCS.items()
        for func in funcs
    ]


# %%
# OBV
FUNCS = {
    'vol': ['OBV'],
}
assert all(func_type in ['price', 'swig', 'vol'] for func_type in FUNCS.keys())
PARAMETER = [
    {}
]
specs += [
        {
            'symbol': 'TXF',
            'type': func_type,
            'freq': '1D',
            'func': func, 
            'args': param
        } 
        for param in PARAMETER
        for func_type, funcs in FUNCS.items()
        for func in funcs
    ]


# %%
# RSI
FUNCS = {
    'swig': ['RSI'],
}
assert all(func_type in ['price', 'swig', 'vol'] for func_type in FUNCS.keys())
PARAMETER = [
    {'timeperiod': 5}, {'timeperiod': 14}, {'timeperiod': 25}
]
specs += [
        {
            'symbol': 'TXF',
            'type': func_type,
            'freq': '1D',
            'func': func, 
            'args': param
        } 
        for param in PARAMETER
        for func_type, funcs in FUNCS.items()
        for func in funcs
    ]


# %%
# SAR
FUNCS = {
    'swig': ['SAR'],
}
assert all(func_type in ['price', 'swig', 'vol'] for func_type in FUNCS.keys())

PARAMETER = [
    {
        'acceleration': 0.02,
        'maximum': 0.2   
    }
]
specs += [
        {
            'symbol': 'TXF',
            'type': func_type,
            'freq': '1D',
            'func': func, 
            'args': param
        } 
        for param in PARAMETER
        for func_type, funcs in FUNCS.items()
        for func in funcs
    ]


# %%
# SMA
FUNCS = {
    'price': ['SMA'],
}
assert all(func_type in ['price', 'swig', 'vol'] for func_type in FUNCS.keys())
PARAMETER = [
    {'timeperiod': 5}, {'timeperiod': 10}, {'timeperiod': 15}, 
         {'timeperiod': 20}, {'timeperiod': 25}, {'timeperiod': 30}
]
specs += [
        {
            'symbol': 'TXF',
            'type': func_type,
            'freq': '1D',
            'func': func, 
            'args': param
        } 
        for param in PARAMETER
        for func_type, funcs in FUNCS.items()
        for func in funcs
    ]


# %%
def get_single_feature(spec):
    df_feature = (
        resampler.get_OHLCV_given_frequency(spec['symbol'], spec['freq']).
        pipe(fg.generate_root_feature, spec=spec)
    )
    return df_feature

def get_featrues(specs, nlags=5):
    df_features = pd.concat([
        get_single_feature(spec).pipe(fg.extend_by_nlags, spec='', nlags=nlags)
        for spec in specs
    ], axis=1)
    
    colnames = []
    for idx_spec in range(len(specs)):
        colnames += list(map(lambda x: f'{idx_spec}_{x}', df_features.columns[idx_spec*nlags: (idx_spec+1)*nlags]))
    df_features.columns = colnames
    
    return df_features


# %%
if __name__ == '__main__':
    nlags=5
    df_features = get_featrues(specs, nlags)
    assert df_features.shape[1] == len(specs) * nlags



# %%
