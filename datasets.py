import collections
import os
from abc import ABC
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.multiprocessing as mp
import torch.utils.data as data
from torch import nn
from tqdm import tqdm


class Interactions(data.Dataset):
    """
    Hold data in the form of an interactions matrix.
    Typical use-case is like a ratings matrix:
    - Users are the rows
    - Items are the columns
    - Elements of the matrix are the ratings given by a user for an item.
    """

    def __init__(self, mat: sp.coo_matrix):
        self.mat = mat.astype(np.float32).tocoo()
        self.mat_csr = self.mat.tocsr()
        self.n_users = self.mat.shape[0]
        self.n_items = self.mat.shape[1]

    def __getitem__(self, index: int) -> Tuple[Tuple[int, int], int]:
        row = self.mat.row[index]
        col = self.mat.col[index]
        val = self.mat.data[index]
        return (row, col), val

    def __len__(self):
        return self.mat.nnz


class PairwiseInteractions(data.Dataset):
    """
    Sample data from an interactions matrix in a pairwise fashion. The row is
    treated as the main dimension, and the columns are sampled pairwise.
    """

    def __init__(self, mat: sp.coo_matrix):
        self.mat = mat.astype(np.float32).tocoo()

        self.n_users = self.mat.shape[0]
        self.n_items = self.mat.shape[1]

        self.mat_csr = self.mat.tocsr()
        if not self.mat_csr.has_sorted_indices:
            self.mat_csr.sort_indices()

    def __getitem__(self, index) -> Tuple[Tuple[int, Tuple[int, int]], int]:
        row = self.mat.row[index]
        found = False

        while not found:
            neg_col = np.random.randint(self.n_items)
            if self.not_rated(row, neg_col, self.mat_csr.indptr, self.mat_csr.indices):
                found = True

        pos_col = self.mat.col[index]
        val = self.mat.data[index]

        return (row, (pos_col, neg_col)), val

    def __len__(self):
        return self.mat.nnz

    @staticmethod
    def not_rated(row: int, col: int, indptr: List[int], indices: List[int]) -> bool:
        # similar to use of bsearch in lightfm
        start = indptr[row]
        end = indptr[row + 1]
        searched = np.searchsorted(indices[start:end], col, "right")
        if searched >= (end - start):
            # After the array
            return False
        return col != indices[searched]  # Not found

    def get_row_indices(self, row: int) -> List[int]:
        start = self.mat_csr.indptr[row]
        end = self.mat_csr.indptr[row + 1]
        return self.mat_csr.indices[start:end]
