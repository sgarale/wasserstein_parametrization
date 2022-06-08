import os
import matplotlib.pyplot as plt
import numpy as np
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

    def __init__(self, func, penalty, h):
        super(PenalisedLoss, self).__init__()
        self.func = func
        self.penalty = penalty
        # self.p = p
        self.h = h

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
        self.penalised_loss = PenalisedLoss(func=func, penalty=penalty,h=h)
        self.mu = mu
        self.nr_sample = nr_sample

    def forward(self):
        y = self.mu.sample([self.nr_sample])
        theta_y = self.theta(y)
        i_theta = self.penalised_loss(y, theta_y)
        return i_theta



def train(model, optimizer):
    model.zero_grad()
    output = model()
    output.backward()
    optimizer.step()
    return output


if __name__ == '__main__':

    torch.manual_seed(29)

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
    torch.set_default_dtype(torch.float64)
    city_center = [0., 0.]
    cost_level = 1.
    radius = 1.5
    epicenter = torch.tensor([1., 0.])
    uncertainty_level = 0.1
    net_width = 20
    mc_samples = 10000
    learning_rate = 0.001
    epochs = 1000

    penalty = pnl.PolyPenaltyTensor(3)
    loss = cf.GaussianKernelCost2DTensor(x_0=city_center[0], y_0=city_center[1],
                                         level=cost_level, radius=radius)
    mu = torch.distributions.multivariate_normal.MultivariateNormal(epicenter, torch.eye(2))

    i_theta = ITheta(func=loss.cost, penalty=penalty.evaluate, mu=mu,
                     h=uncertainty_level, width=net_width, nr_sample=mc_samples)

    # computing the expected loss via montecarlo
    rnd_y = mu.sample([1000000])
    expected_loss = torch.mean(loss.cost(rnd_y[:, 0], rnd_y[:, 1]))
    print("-------------------------------------")
    print(f"Expected Loss: {expected_loss:.4f}")
    optm = torch.optim.Adam(i_theta.parameters(), lr=learning_rate)
    out_vector = []
    for i in range(epochs):
        out = train(i_theta, optm)
        out_vector.append(-float(out))
    print(f"Parametrized worst case loss: {-out:.4f}")
    plt.plot(np.arange(1, epochs + 1), out_vector)
    plt.title(f"Training of the worst case loss for h={uncertainty_level}")
    plt.xlabel("Epochs")
    plt.ylabel("Worst case loss")
    plt.show()

    # Drawing the vector field theta
    x = np.arange(-1.7, 1.8, 0.1)
    y = np.arange(-1.7, 1.8, 0.1)
    xv, yv = np.meshgrid(x, y)
    theta = i_theta.theta(torch.stack([torch.from_numpy(xv), torch.from_numpy(yv)], dim=2)).detach().numpy()

    # Depict illustration
    plt.figure(figsize=(10, 10))
    plt.streamplot(xv, yv, theta[:, :, 0], theta[:, :, 1], density=1.4, linewidth=None, color='#A23BEC')
    plt.title('Direction of the optimization')
    plt.show()
