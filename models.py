import os
import sys
from itertools import islice
from math import ceil
from subprocess import call
from typing import Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy
import scipy.sparse as sp
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scipy.sparse import csr_matrix, dok_matrix
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import normalize
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange


class MFModule(nn.Module):
    """
    Base module for explicit matrix factorization.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 40,
        dropout_p: float = 0,
        sparse: bool = False,
    ):
        """

        Parameters
        ----------
        n_factors : int
            Number of latent factors (or embeddings or whatever you want to
            call it).
        dropout_p : float
            p in nn.Dropout module. Probability of dropout.
        sparse : bool
            Whether or not to treat embeddings as sparse. NOTE: cannot use
            weight decay on the optimizer if sparse=True. Also, can only use
            Adagrad.
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_biases = nn.Embedding(n_users, 1, sparse=sparse)
        self.item_biases = nn.Embedding(n_items, 1, sparse=sparse)
        self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(n_items, n_factors, sparse=sparse)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.sparse = sparse

    def forward(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """
        Forward pass through the model. For a single user and item, this
        looks like:

        user_bias + item_bias + user_embeddings.dot(item_embeddings)

        Parameters
        ----------
        users : np.ndarray
            Array of user indices
        items : np.ndarray
            Array of item indices

        Returns
        -------
        preds : np.ndarray
            Predicted ratings.

        """
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)

        preds = self.user_biases(users)
        preds += self.item_biases(items)
        preds += (self.dropout(ues) * self.dropout(uis)).sum(dim=1, keepdim=True)

        return preds.squeeze()

    def __call__(self, *args) -> np.ndarray:
        return self.forward(*args)

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return self.forward(users, items)


def bpr_loss(preds, _) -> torch.Tensor:
    sig = nn.Sigmoid()
    return (1.0 - sig(preds)).pow(2).sum()


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


class BPRModule(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 40,
        dropout_p: float = 0,
        sparse: bool = False,
        model: nn.Module = MFModule,
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.dropout_p = dropout_p
        self.sparse = sparse
        self.pred_model = model(
            self.n_users,
            self.n_items,
            n_factors=n_factors,
            dropout_p=dropout_p,
            sparse=sparse,
        )

    def forward(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        # assert isinstance(items, tuple), "Must pass in items as (pos_items, neg_items)"
        # Unpack
        (pos_items, neg_items) = items
        pos_preds = self.pred_model(users, pos_items)
        neg_preds = self.pred_model(users, neg_items)
        return pos_preds - neg_preds

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return self.pred_model(users, items)


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

    def validation_epoch_end(self, outputs):
        aucs = []
        for output in outputs:
            aucs.append(batch_auc(output["users"], self.csr_mat, self.model))
        self.log("val_roc", torch.Tensor([np.mean(aucs)]), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


def get_row_indices(row, interactions):
    start = interactions.indptr[row]
    end = interactions.indptr[row + 1]
    return interactions.indices[start:end]


class BPR:
    """
    References
    ----------
    S. Rendle, C. Freudenthaler, Z. Gantner, L. Schmidt-Thieme
    Bayesian Personalized Ranking from Implicit Feedback
    - https://arxiv.org/abs/1205.2618
    """

    def __init__(
        self,
        learning_rate=0.01,
        n_factors=15,
        n_iters=10,
        batch_size=1000,
        reg=0.01,
        seed=1234,
        verbose=True,
    ):
        self.reg = reg
        self.seed = seed
        self.verbose = verbose
        self.n_iters = n_iters
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # to avoid re-computation at predict
        self._prediction = None

    def fit(self, ratings: csr_matrix):
        """
        Parameters
        ----------
        ratings : scipy sparse csr_matrix, shape [n_users, n_items]
            sparse matrix of user-item interactions
        """
        indptr = ratings.indptr
        indices = ratings.indices
        n_users, n_items = ratings.shape

        batch_size = self.batch_size
        if n_users < batch_size:
            batch_size = n_users
            sys.stderr.write(
                "WARNING: Batch size is greater than number of users,"
                "switching to a batch size of {}\n".format(n_users)
            )

        batch_iters = n_users // batch_size

        # initialize random weights
        rstate = np.random.RandomState(self.seed)
        self.user_factors = rstate.normal(size=(n_users, self.n_factors))
        self.item_factors = rstate.normal(size=(n_items, self.n_factors))

        # progress bar for training iteration if verbose is turned on
        loop = range(self.n_iters)
        if self.verbose:
            loop = trange(self.n_iters, desc=self.__class__.__name__)

        for _ in loop:
            for _ in range(batch_iters):
                sampled = self._sample(n_users, n_items, indices, indptr)
                sampled_users, sampled_pos_items, sampled_neg_items = sampled
                self._update(sampled_users, sampled_pos_items, sampled_neg_items)

        return self

    def _sample(self, n_users, n_items, indices, indptr):
        """sample batches of random triplets u, i, j"""
        sampled_pos_items = np.zeros(self.batch_size, dtype=int)
        sampled_neg_items = np.zeros(self.batch_size, dtype=int)
        sampled_users = np.random.choice(n_users, size=self.batch_size, replace=False)

        for idx, user in enumerate(sampled_users):
            pos_items = indices[indptr[user] : indptr[user + 1]]
            pos_item = np.random.choice(pos_items)
            neg_item = np.random.choice(n_items)
            while neg_item in pos_items:
                neg_item = np.random.choice(n_items)

            sampled_pos_items[idx] = pos_item
            sampled_neg_items[idx] = neg_item

        return sampled_users, sampled_pos_items, sampled_neg_items

    def _update(self, u, i, j):
        """
        update according to the bootstrapped user u,
        positive item i and negative item j
        """
        user_u = self.user_factors[u]
        item_i = self.item_factors[i]
        item_j = self.item_factors[j]

        # ドット積を行った後対角線上の要素をだけを取り出すのではなく、
        # 行列の要素ごとの積(ハダマート積)をとった後、列に沿った和を取ることで高速化
        # さらに差分をとってから、行列積を取ることで効率化
        r_uij = np.sum(user_u * (item_i - item_j), axis=1)
        sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))

        # repeat the 1 dimension sigmoid n_factors times so
        # the dimension will match when doing the update
        sigmoid_tiled = np.tile(sigmoid, (self.n_factors, 1)).T

        # update using gradient descent
        grad_u = sigmoid_tiled * (item_j - item_i) + self.reg * user_u
        grad_i = sigmoid_tiled * -user_u + self.reg * item_i
        grad_j = sigmoid_tiled * user_u + self.reg * item_j
        self.user_factors[u] -= self.learning_rate * grad_u
        self.item_factors[i] -= self.learning_rate * grad_i
        self.item_factors[j] -= self.learning_rate * grad_j
        return self

    def _predict_user(self, user):
        """
        returns the predicted ratings for the specified user,
        this is mainly used in computing evaluation metric
        """
        user_pred = self.user_factors[user].dot(self.item_factors.T)
        return user_pred

    def recommend(self, ratings, N=5):
        """
        Returns the top N ranked items for given user id,
        excluding the ones that the user already liked

        Parameters
        ----------
        ratings : scipy sparse csr_matrix, shape [n_users, n_items]
            sparse matrix of user-item interactions

        N : int, default 5
            top-N similar items' N

        Returns
        -------
        recommendation : 2d ndarray, shape [number of users, N]
            each row is the top-N ranked item for each query user
        """
        n_users = ratings.shape[0]
        recommendation = np.zeros((n_users, N), dtype=np.uint32)
        for user in range(n_users):
            top_n = self._recommend_user(ratings, user, N)
            recommendation[user] = top_n

        return recommendation

    def _recommend_user(self, ratings, user, N):
        """the top-N ranked items for a given user"""
        scores = self._predict_user(user)

        # compute the top N items, removing the items that the user already liked
        # from the result and ensure that we don't get out of bounds error when
        # we ask for more recommendations than that are available
        liked = set(ratings[user].indices)
        count = N + len(liked)
        if count < scores.shape[0]:

            # when trying to obtain the top-N indices from the score,
            # using argpartition to retrieve the top-N indices in
            # unsorted order and then sort them will be faster than doing
            # straight up argort on the entire score
            # http://stackoverflow.com/questions/42184499/cannot-understand-numpy-argpartition-output
            ids = np.argpartition(scores, -count)[-count:]
            best_ids = np.argsort(scores[ids])[::-1]
            best = ids[best_ids]
        else:
            best = np.argsort(scores)[::-1]

        top_n = list(islice((rec for rec in best if rec not in liked), N))
        return top_n
