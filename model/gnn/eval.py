import numpy as np

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy().reshape(-1)

    return np.sum(y_true == y_pred) / len(y_true)

def eval_acc_for_class(y_true, y_pred, num_class):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy().reshape(-1)
    acc_for_class = {}
    for i in range(num_class):
        acc_for_class[i] = np.sum((y_true == y_pred)[np.where(y_true == i)[0]]) / np.sum(y_true == i)

        
    return acc_for_class