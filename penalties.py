import numpy as np
import torch


class Penalty:

    def __init__(self):
        self.penal_type = None

    def evaluate(self, x):
        raise NotImplemented('Method "evaluate" must be implemented.')



class PolyPenalty(Penalty):

    def __init__(self, p):
        super().__init__()
        if p <= 1:
            raise Exception(f"Polynomial penalty function can be defined only for powers greater than 1. Passed {p: .4f} instead.")
        self.penal_type = "Polynomial penalty"
        self.p = p

    def evaluate(self, x):
        if x < 0:
            raise Exception(f"Penalty function is defined only for positive numbers. Passed {x:.4f} instead.")
        return np.power(x, self.p)



class PolyPenaltyTensor(PolyPenalty):

    def __init__(self, p):
        super().__init__(p)
        self.penal_type = "Polynomial penalty (tensorial)"

    def evaluate(self, x):
        if x < 0:
            raise Exception(f"Penalty function is defined only for positive numbers. Passed {x:.4f} instead.")
        return torch.pow(x, self.p)


class RescaledPolyPenalty(Penalty):

    def __init__(self, sigma, n):
        super().__init__()
        self.penal_type = 'Rescaled polynomial penalty'
        self.sigma = torch.tensor(sigma)
        self.sigma2 = torch.tensor(sigma * sigma)
        self.n = torch.tensor(n)

    def evaluate(self, x):
        return torch.pow(x / self.sigma2, self.n) / self.n