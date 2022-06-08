import numpy as np
import torch


class Penalty:

    def __init__(self):
        self.penal_type = None


class PolyPenaltyTensor(Penalty):

    def __init__(self, p):
        super().__init__()
        if p <= 1:
            raise Exception(f"Polynomial penalty function is defined only for powers greater than 1. Passed {p: .4f} instead.")
        self.penal_type = "Polynomial penalty (tensorial)"
        self.p = p

    def evaluate(self, x):
        if x < 0:
            raise Exception(f"Penalty function is defined only for positive numbers. Passed {x: .4f} instead.")
        return torch.pow(x, self.p)
