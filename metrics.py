import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from utils import get_row_indices


class Metrics:
    def __init__(self, k: int) -> None:
        self.k = k
        self.metrics_list = []

    def append(self, pred: np.ndarray, test: np.ndarray):
        self.metrics_list.append(self._core(pred, test))

    def mean(self) -> float:
        return np.mean(self.metrics_list)

    def _core(self, pred: np.ndarray, test: np.ndarray):
        pass


class Recall(Metrics):
    def _core(self, pred: np.ndarray, test: np.ndarray):
        k = int(min(np.sum(test), self.k))
        arg_idx = np.argsort(pred)[::-1][:k]
        return np.sum(test[arg_idx]) / k


class Precision(Metrics):
    def _core(self, pred: np.ndarray, test: np.ndarray):
        k = int(min(np.sum(test), self.k))
        arg_idx = np.argsort(pred)[::-1][:k]
        return np.sum(test[arg_idx]) / len(arg_idx)


class MRR(Metrics):
    def _core(self, pred: np.ndarray, test: np.ndarray):
        k = int(min(np.sum(test), self.k))
        arg_idx = np.argsort(pred)[::-1][:k]
        nonzero_idx = test[arg_idx].nonzero()[0]

        if nonzero_idx.size == 0:
            return 0

        return 1 / (nonzero_idx[0] + 1)


def batch_auc(rows: torch.Tensor, interactions: sp.csr_matrix, model) -> float:

    n_items = interactions.shape[1]
    items = torch.arange(0, n_items).long()
    users_init = torch.ones(n_items).long()

    aucs = []
    for row in rows:
        row = int(row)
        users = users_init.fill_(row)

        preds = model.predict(users, items)
        actuals = get_row_indices(row, interactions)

        if len(actuals) == 0:
            continue

        y_test = np.zeros(n_items)
        y_test[actuals] = 1
        aucs.append(roc_auc_score(y_test, preds.data.numpy()))
    return np.mean(aucs)
