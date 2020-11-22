#!/usr/bin/env python
# coding: utf-8

# In[1]:


from findataflow.dataprocs import feature_generator as fg
from findataflow.dataprocs import resample as resampler


# In[2]:


import pandas as pd


# In[3]:


specs = []


# In[4]:


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


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


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


# In[10]:


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


# In[11]:


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


# In[12]:


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


# In[13]:


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


# In[14]:


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


# In[15]:


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


# In[20]:


def get_single_feature(spec):
    df_feature = resampler.get_OHLCV_given_frequency(spec['symbol'], spec['freq']).pipe(fg.generate_root_feature, spec=spec)
    return df_feature

def get_featrues(specs):
    df_features = pd.concat([get_single_feature(spec) for spec in specs], axis=1)
    return df_features


# In[ ]:

if __name__ == '__main__':
    df_features = get_featrues(specs)
    df_features.shape
    assert df_features.shape[1] == len(specs)

