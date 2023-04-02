import functools
import optuna
import catboost as cbt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error


def objective(pools, loss, trial):
    params = {
        'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.01, 0.8),
        'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 10, 800),
        'depth': trial.suggest_int('depth', 1, 10),
        'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 0.5),
        'loss_function': loss,
        'random_seed': 0,
        'task_type': 'CPU'
    }
    
    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
    elif params['bootstrap_type'] == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.1, 1)

    if params['loss_function'] == 'Huber':
        # loss_function='Huber:delta=0.5'
        params['loss_function'] = f'Huber:delta={trial.suggest_float("delta", 0.01, 1.0)}'
        # params['delta'] = trial.suggest_float("delta", 0.01, 1.0),
    
    model = cbt.CatBoostRegressor(**params,iterations = 5000,)

    model.fit(
        pools['train'],
        eval_set=[pools['valid'], pools['test']],
        use_best_model=True,
        plot=False,
        verbose=50,
        early_stopping_rounds=50,
    )

    return list(model.best_score_['validation_0'].values())[0]