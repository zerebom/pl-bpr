import torch
from torch import nn


def bpr_loss(preds, _) -> torch.Tensor:
    sig = nn.Sigmoid()
    return (1.0 - sig(preds)).pow(2).sum()
