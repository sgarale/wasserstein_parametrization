import os
import numpy as np
import torch
from torch import nn


class ThetaSigmoidNetwork(nn.Module):
    def __init__(self, depth, width):
        super(ThetaSigmoidNetwork, self).__init__()
        self.linear_sigm_stack = nn.Sequential(
            nn.Linear(2, width),
            nn.Sigmoid(),
            nn.Linear(width, 2)
        )

    def forward(self, x):
        theta = self.linear_sigm_stack(x)
        return theta


class PenalisedLoss(nn.Module):
    def __init__(self, func, penalty, mu, h, nr_samples):
        self.func = func
        self.penalty = penalty
        self.mu = mu
        # self.p = p
        self.h = h
        self.nr_samples = nr_samples


    def forward(self, theta):
        y = self.mu.rand(self.nr_samples, 2)
        theta_y = theta(y)
        # integral = torch.mean(self.func(y[:, 0] + theta_y[:, 0], y[:, 1] + theta_y[:, 1]) * self.mu.density(y[:, 0], y[:, 1]))
        integral = torch.mean(self.func(y[:, 0] + theta_y[:, 0], y[:, 1] + theta_y[:, 1]))
        # L2_norm_theta = torch.mean(torch.pow(theta_y, 2).sum(dim=1) * self.mu.density(y[:, 0], y[:, 1])).sqrt()
        L2_norm_theta = torch.mean(torch.pow(theta_y, 2).sum(dim=1)).sqrt()
        # Lp_norm_theta = torch.mean(torch.pow(torch.linalg.vector_norm(theta_y, ord=self.p, dim=1), self.p)).sqrt()
        penal_term = self.h * self.penalty(L2_norm_theta / self.h)
        return - (integral - self.h * penal_term)


def train(model, x, optimizer):
    model.zero_grad()
    output = model(x)
    output.backward()
    optimizer.step()
    return output


if __name__ == '__main__':

    # np.random.seed(0)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    theta = ThetaSigmoidNetwork(1, 5).to(device)
    print(theta)

    # let's see the parameters of the model
    print("-----------------------------------------------")
    for name, param in theta.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param} \n")

    print("-----------------------------------------------")
    x = torch.tensor([1., 2.])
    # x = torch.rand(10, 2, device=device)
    print(f"random initial points: {x}")
    directions = theta(x)
    print(directions)

    x = torch.tensor([[1., 2.], [3., 4.], [5, 6]])
    # x = torch.rand(10, 2, device=device)
    print(f"evaluation points: {x}")
    directions = theta(x)
    print(directions)

    exit()
    EPOCHS = 200
    optm = torch.optim.Adam(theta.parameters(), lr=0.001)