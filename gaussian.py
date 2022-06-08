import os
import numpy as np
from scipy.stats import multivariate_normal
import torch
from torch import nn
import cost_functions as cf
import penalties as pnl


class Gaussian2D:

    def __init__(self, mu_x, mu_y, covariance_matrix):
        self.multinormal = torch.tensor([mu_x, mu_y])


    def sample(self, size):
        rnd = torch.normal(mean = 0, std=1, size=(size, 2))
        rnd[:, 1] = self.rho * rnd[:, 0] + torch.sqrt(1. - torch.pow(self.rho, 2)) * rnd[:, 1]
        return rnd


class ThetaSigmoidNetwork(nn.Module):

    def __init__(self, width):
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
        super(PenalisedLoss, self).__init__()
        self.func = func
        self.penalty = penalty
        self.mu = mu
        # self.p = p
        self.h = h
        self.nr_samples = nr_samples

    def forward(self, y, theta_y):
        integral = torch.mean(self.func(y[:, 0] + theta_y[:, 0], y[:, 1] + theta_y[:, 1]))
        L2_norm_theta = torch.mean(torch.pow(theta_y, 2).sum(dim=1)).sqrt()
        penal_term = self.h * self.penalty(L2_norm_theta / self.h)
        return - (integral - self.h * penal_term)

    def forward_old(self, theta):
        y = self.mu.rand(self.nr_samples, 2)
        theta_y = theta(y)
        # integral = torch.mean(self.func(y[:, 0] + theta_y[:, 0], y[:, 1] + theta_y[:, 1]) * self.mu.density(y[:, 0], y[:, 1]))
        integral = torch.mean(self.func(y[:, 0] + theta_y[:, 0], y[:, 1] + theta_y[:, 1]))
        # L2_norm_theta = torch.mean(torch.pow(theta_y, 2).sum(dim=1) * self.mu.density(y[:, 0], y[:, 1])).sqrt()
        L2_norm_theta = torch.mean(torch.pow(theta_y, 2).sum(dim=1)).sqrt()
        # Lp_norm_theta = torch.mean(torch.pow(torch.linalg.vector_norm(theta_y, ord=self.p, dim=1), self.p)).sqrt()
        penal_term = self.h * self.penalty(L2_norm_theta / self.h)
        return - (integral - self.h * penal_term)



class ITheta(nn.Module):

    def __init__(self, func, penalty, mu, h, width, nr_sample):
        super(ITheta, self).__init__()
        self.theta = ThetaSigmoidNetwork(width)
        self.penalised_loss = PenalisedLoss(func=func, penalty=penalty, mu=mu, h=h, nr_samples=nr_sample)
        self.nr_sample = nr_sample

    def forward(self):
        y = self.mu.sample([self.nr_samples])
        theta_y = self.theta(y)
        i_theta = self.penalised_loss(y, theta_y)
        return i_theta



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

    theta = ThetaSigmoidNetwork(5).to(device)
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

    #################### ITheta tests #####################
    city_center = [0., 0.]
    cost_level = 1.
    radius = 1.5
    net_width = 10
    mc_samples = 1000

    penalty = pnl.PolyPenaltyTensor(3)
    loss = cf.GaussianKernelCost2DTensor(x_0=city_center[0], y_0=city_center[1],
                                         level=cost_level, radius=radius)
    mu = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.eye(2))

    i_theta = ITheta(func=loss, penalty=penalty, mu=mu, h=0.1, width=10, nr_sample=mc_samples)

    EPOCHS = 200
    optm = torch.optim.Adam(i_theta.parameters(), lr=0.001)
