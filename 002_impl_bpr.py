#%%
import os
import sys
from itertools import islice
from math import ceil
from subprocess import call
from typing import Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.sparse as sp
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from scipy.sparse import csr_matrix, dok_matrix
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import normalize
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

import utils
from datasets import Interactions, PairwiseInteractions
from metrics import auc
from models import BPR, BPRModule, MFModule, MFPLModule
from torchmf import bpr_loss

batch_size = 1024
num_workers = 0

train_data, test_data = utils.get_movielens_train_test_split(implicit=True)

train_pair_interactions = PairwiseInteractions(train_data)
test_pair_interactions = PairwiseInteractions(test_data)


train_pair_loader = DataLoader(
    train_pair_interactions,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)
test_pair_loader = DataLoader(
    test_pair_interactions,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)


bpr_pl_module = MFPLModule(train_pair_interactions.mat_csr, model=BPRModule)


early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min"
)

wandb_logger = WandbLogger()

n_epochs = 500
trainer = pl.Trainer(
    gpus=0,
    max_epochs=n_epochs,
    checkpoint_callback=False,
    logger=wandb_logger,
    callbacks=[early_stop_callback],
)


trainer.fit(
    bpr_pl_module, train_dataloader=train_pair_loader, val_dataloaders=test_pair_loader
)


# %%
preds = (
    bpr_pl_module.model.predict(
        torch.Tensor(test_data.row).to(int), torch.Tensor(test_data.col).to(int)
    )
    .detach()
    .numpy()
)
#%%
# to_dense
pred_matrix = np.zeros(test_data.shape)
for i in range(len(test_data.row)):
    pred_matrix[test_data.row[i], test_data.col[i]] = preds[i]

#%%
preds2 = (
    base_pl_module.model.predict(
        torch.Tensor(test.row).to(int), torch.Tensor(test.col).to(int)
    )
    .detach()
    .numpy()
)

#%%
def get_row_indices(row: int, interactions: sp.csr_matrix) -> np.ndarray:
    start = interactions.indptr[row]
    end = interactions.indptr[row + 1]
    return interactions.indices[start:end]


def batch_auc(
    rows: torch.Tensor, interactions: sp.csr_matrix, model: nn.Module
) -> float:

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


#%%


#%%
n_users, n_items = train_data.shape
bpr = BPRModule(n_users, n_items)

(users, pair), val = batch_input
preds = bpr(users, pair)
loss = bpr_loss(preds)

batch_auc(users, train_pair_interactions.mat_csr, bpr)

#%%
def predict(model, test_loader, shape):
    preds = np.zeros(shape)
    for (b_rows, b_cols), b_vals in test_loader:
        b_preds = model.model.predict(b_rows, b_cols[0]).detach().numpy()
        b_rows = b_rows.detach().numpy()
        b_cols = b_cols.detach().numpy()
        for row, col, pred in zip(b_rows, b_cols, b_preds):
            preds[row, col] = pred
        return preds
