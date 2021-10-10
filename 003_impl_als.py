#%%
# %load_ext autoreload
# %autoreload 2


from typing import Callable, NamedTuple, Optional, TypedDict

import numpy as np
import pytorch_lightning as pl
import scipy.sparse as sp
import torch
from pytorch_lightning.loggers import WandbLogger
from scipy.sparse import coo_matrix, csr_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from datasets import Interactions, PairwiseInteractions
from metrics import get_row_indices
from models import BPRModule, MFModule, MFPLModule, bpr_loss

train_data, test_data = utils.get_movielens_train_test_split(implicit=True)

# %%


class Params(TypedDict):
    batch_size: Optional[int]
    num_workers: int
    n_epochs: Optional[int]


params = Params(batch_size=1024, num_workers=0, n_epochs=50)


#%%


def train(
    train_data: sp.coo.coo_matrix,
    test_data: sp.coo.coo_matrix,
    model=MFModule,
    datasets_cls=Interactions,
    loss: Callable = nn.MSELoss(reduction="sum"),
    params: Params = params,
) -> MFPLModule:
    train_dataset = datasets_cls(train_data)
    test_dataset = datasets_cls(test_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
    )

    pl_module = MFPLModule(train_dataset.mat_csr, model=model, loss=loss)

    wandb_logger = WandbLogger()
    trainer = pl.Trainer(
        gpus=0,
        max_epochs=params["n_epochs"],
        checkpoint_callback=False,
        logger=wandb_logger,
    )

    trainer.fit(pl_module, train_dataloader=train_loader, val_dataloaders=test_loader)
    return pl_module


#%%
mf_module = train(
    train_data,
    test_data,
    model=MFModule,
    datasets_cls=Interactions,
    loss=nn.MSELoss(reduction="sum"),
    params=params,
)

#%%

bpr_module = train(
    train_data,
    test_data,
    model=BPRModule,
    datasets_cls=PairwiseInteractions,
    loss=bpr_loss,
    params=params,
)

#%%


def pred_by_user(user: int, interactions: sp.csr_matrix, model: nn.Module):
    n_items = interactions.shape[1]
    items = torch.arange(0, n_items).long()
    users = torch.ones(n_items).long().fill_(user)

    preds = model.predict(users, items).detach().numpy()
    actuals = get_row_indices(user, interactions)

    if len(actuals) == 0:
        return None, None

    y_test = np.zeros(n_items)
    y_test[actuals] = 1

    return preds, y_test


#%%


class Metrics:
    def __init__(self, k):
        self.k = k
        self.metrics_list = []

    def append(self, pred, test):
        self.metrics_list.append(self._core(pred, test))

    def mean(self):
        return np.mean(self.metrics_list)

    def _core(self, pred, test):
        pass



class Recall(Metrics):
    def _core(self, pred, test):
        k = min(np.sum(test), self.k)
        arg_idx = np.argsort(pred)[::-1][:k]
        return np.sum(test[arg_idx]) / k


class Precision(Metrics):
    def _core(self, pred, test):
        k = min(np.sum(test), self.k)
        arg_idx = np.argsort(pred)[::-1][:k]
        return np.sum(test[arg_idx]) / len(arg_idx)


class MRR(Metrics):
    def _core(self, pred, test):
        k = min(np.sum(test), self.k)
        arg_idx = np.argsort(pred)[::-1][:k]
        nonzero_idx = test[arg_idx].nonzero()[0]

        if nonzero_idx.size == 0:
            return 0

        print(nonzero_idx[0])
        return 1 / (nonzero_idx[0] + 1)

recall = Recall(k=30)
precision = Precision(k=30)
mrr = MRR(k=30)

metrics = [recall, precision, mrr]

n_users = test_data.shape[0]
for user_id in tqdm(range(n_users)):
    pred, test = pred_by_user(user_id, test_data.tocsr(), bpr_module.model)
    if pred is None:
        continue

    for met in metrics:
        met.append(pred, test)

[met.mean() for met in metrics]

