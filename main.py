from typing import Callable, Optional, TypedDict

import numpy as np
import pytorch_lightning as pl
import scipy.sparse as sp
import torch
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from datasets import Interactions, PairwiseInteractions
from losses import bpr_loss
from metrics import MRR, Precision, Recall, get_row_indices
from models import BPRModule, MFModule
from pl_modules import MFPLModule


class Params(TypedDict):
    batch_size: Optional[int]
    num_workers: int
    n_epochs: Optional[int]


def train(
    train_data: sp.coo.coo_matrix,
    test_data: sp.coo.coo_matrix,
    model,
    datasets_cls,
    params: Params,
    loss: Callable,
) -> MFPLModule:

    train_ds = datasets_cls(train_data)
    test_ds = datasets_cls(test_data)

    train_dl = DataLoader(
        train_ds,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
    )

    pl_module = MFPLModule(train_ds.mat_csr, model=model, loss=loss)
    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=params["n_epochs"],
        checkpoint_callback=False,
        logger=wandb_logger,
    )

    trainer.fit(pl_module, train_dataloader=train_dl, val_dataloaders=test_dl)
    return pl_module


def evaluate(model, test_data: sp.csr_matrix):
    def pred_by_user(user: int, interactions: sp.csr_matrix, model):
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

    metrics = [Recall(k=30), Precision(k=30), MRR(k=30)]

    n_users = test_data.shape[0]
    for user_id in tqdm(range(n_users)):
        pred, test = pred_by_user(user_id, test_data.tocsr(), model)
        if pred is None:
            continue

        for metric in metrics:
            metric.append(pred, test)

    return [m.mean() for m in metrics]


def run():
    train_data, test_data = utils.get_movielens_train_test_split(implicit=True)
    params = Params(batch_size=1024, num_workers=0, n_epochs=50)

    mf_module = train(
        train_data,
        test_data,
        model=MFModule,
        datasets_cls=Interactions,
        loss=nn.MSELoss(reduction="sum"),
        params=params,
    )

    bpr_module = train(
        train_data,
        test_data,
        model=BPRModule,
        datasets_cls=PairwiseInteractions,
        loss=bpr_loss,
        params=params,
    )

    print(evaluate(mf_module.model, test_data.tocsr()))
    print(evaluate(bpr_module.model, test_data.tocsr()))


if __name__ == "__main__":
    run()
