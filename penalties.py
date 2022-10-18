import numpy as np
import torch


class Penalty:

    def __init__(self):
        self.penal_type = None



class PolyPenalty(Penalty):

    def __init__(self, p):
        super().__init__()
        if p <= 1:
            raise Exception(f"Polynomial penalty function is defined only for powers greater than 1. Passed {p: .4f} instead.")
        self.penal_type = "Polynomial penalty"
        self.p = p

    def evaluate(self, x):
        if x < 0:
            raise Exception(f"Penalty function is defined only for positive numbers. Passed {x: .4f} instead.")
        return np.power(x, self.p)



class PolyPenaltyTensor(PolyPenalty):

    def __init__(self, p):
        super().__init__(p)
        self.penal_type = "Polynomial penalty (tensorial)"

    def evaluate(self, x):
        if x < 0:
            raise Exception(f"Penalty function is defined only for positive numbers. Passed {x: .4f} instead.")
        return torch.pow(x, self.p)



class InfinitePenalty(Penalty):

    def __init__(self, level):
        super().__init__()
        if level <= 0:
            raise Exception(f'Infinite penalty level must be positive. Passed {level:.4f}')
        self.penal_type = 'Infinite penalty'
        self.level = level

    def evaluate(self, x):
        if x > self.level:
            return torch.tensor(float('Inf'))
        return torch.tensor(0.)


class PowerGrowthPenalty(Penalty):

    def __init__(self, scaling, steepness, denominator=1):
        super().__init__()
        if scaling <= 0:
            raise Exception(f'Scaling must be bigger than 0')
        if steepness <= 1:
            raise Exception(f'Power growth must be bigger than 1')
        self.penal_type = 'power growth penalty'
        self.scaling = torch.tensor(scaling)
        self.steepness = torch.tensor(steepness)
        self.denominator = torch.tensor(denominator)

    def evaluate(self, x):
        cost = torch.pow(self.scaling * x, self.steepness) / self.denominator
        return cost