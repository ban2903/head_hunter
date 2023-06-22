from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    print(logits)
    print('~'*10)
    print(labels)
    labels = labels.reshape(-1, 1)
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    
    return {"mse": mse, "mae": mae, "r2": r2}