import numpy as np
import pytorch_lightning as pl
import torch

from losses import bpr_loss
from metrics import batch_auc
from models import MFModule


class MFPLModule(pl.LightningModule):
    def __init__(
        self,
        csr_mat,
        n_factors=10,
        lr=0.02,
        dropout_p=0.02,
        weight_decay=0.1,
        model=MFModule,
        loss=bpr_loss,
    ):
        super().__init__()
        self.csr_mat = csr_mat
        self.n_users = csr_mat.shape[0]
        self.n_items = csr_mat.shape[1]

        self.n_factors = n_factors
        self.dropout_p = dropout_p
        self.lr = lr
        self.loss = loss
        self.weight_decay = weight_decay
        self.model = model(
            self.n_users,
            self.n_items,
            n_factors=self.n_factors,
            dropout_p=self.dropout_p,
            sparse=False,
        )

    def forward(self, users: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        return self.model(users, pairs)

    def training_step(self, batch, batch_idx):
        (users, paris), vals = batch
        preds = self.model(users, paris)
        loss = self.loss(preds, vals)

        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        (users, paris), vals = batch
        preds = self.model(users, paris)

        loss = self.loss(preds, vals)
        self.log("val_loss", loss, prog_bar=False)
        return {"users": users, "preds": preds, "loss": loss}

    # def validation_epoch_end(self, outputs):
    # TODO: batch_auc is too slow. We should use a faster metric.
    # aucs = []
    # for output in outputs:
    #     aucs.append(batch_auc(output["users"], self.csr_mat, self.model))
    # self.log("val_roc", torch.Tensor([np.mean(aucs)]), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
