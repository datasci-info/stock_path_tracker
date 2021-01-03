# stock_path_tracker

## environment
After cloning this repository, please also add submodule:
- datasci-info/fflowlab
Then please build the conda environment with the following codes:
```
conda env create -f environment.yml
conda activate stock_path_tracker
```


## workflow
- data
    - turnpt_analysis.py  : y
    - prepare_X.py        : X
    - data_ML.py          : data preparation
- signal
    - model_logistic_regression.py : logistic regression
    - catboost_model               : catboost classification
        - catboost_hyperpara_searching.py
    - signal_confirmation.py
- strategy
    - strategy_nav.py