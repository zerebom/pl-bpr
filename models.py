import numpy as np
from torch import nn


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


class BPRModule(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 40,
        dropout_p: float = 0,
        sparse: bool = False,
        model=MFModule,
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
