import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import cost_functions as cf
import penalties as pnl


class Gaussian2D:
    """
    Generator of random points for a two dimensional Gaussian measure.
    """
    def __init__(self, mean, covariance_matrix):
        self.multinormal = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor(mean),
                                                                                      torch.tensor(covariance_matrix))
        self.mean = np.array(mean)
        self.covariance_matrix = np.array(covariance_matrix)

    def sample(self, size):
        return self.multinormal.sample(size)

    def density(self, x):
        x_m = x - self.mean
        return (1.0 / (np.sqrt((2 * np.pi) ** 2 * np.linalg.det(self.covariance_matrix))) * np.exp(
            -(np.linalg.solve(self.covariance_matrix, x_m).T.dot(x_m)) / 2))


class ThetaReLUNetwork(nn.Module):

    def __init__(self, width, depth=1):
        super(ThetaReLUNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, width)
        )
        for i in range(depth - 1):
            self.linear_relu_stack.append(nn.ReLU())
            self.linear_relu_stack.append(nn.Linear(width, width))
        self.linear_relu_stack.append(nn.ReLU())
        self.linear_relu_stack.append(nn.Linear(width, 2))

    def forward(self, x):
        theta = self.linear_relu_stack(x)
        return theta

class ThetaSigmoidNetwork(nn.Module):

    def __init__(self, width, depth=1):
        super(ThetaSigmoidNetwork, self).__init__()
        self.linear_sigm_stack = nn.Sequential(
            nn.Linear(2, width)
        )
        for i in range(depth - 1):
            self.linear_sigm_stack.append(nn.Sigmoid())
            self.linear_sigm_stack.append(nn.Linear(width, width))
        self.linear_sigm_stack.append(nn.Sigmoid())
        self.linear_sigm_stack.append(nn.Linear(width, 2))

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



class ITheta(nn.Module):

    def __init__(self, func, penalty, mu, h, width, depth, nr_sample):
        super(ITheta, self).__init__()
        self.theta = ThetaReLUNetwork(width, depth)
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
    net_depth = 1
    mc_samples = 10000
    learning_rate = 0.001
    epochs = 1000

    penalty = pnl.PolyPenaltyTensor(3)
    loss = cf.GaussianKernelCost2DTensor(x_0=city_center[0], y_0=city_center[1],
                                         level=cost_level, radius=radius)
    mu = torch.distributions.multivariate_normal.MultivariateNormal(epicenter, torch.eye(2))
    # mu = Gaussian2D([1., 0.], np.eye(2))
    i_theta = ITheta(func=loss.cost, penalty=penalty.evaluate, mu=mu,
                     h=uncertainty_level, width=net_width, depth=net_depth,
                     nr_sample=mc_samples)

    # computing the expected loss via montecarlo
    rnd_y = mu.sample([1000000])
    expected_loss = torch.mean(loss.cost(rnd_y[:, 0], rnd_y[:, 1]))
    print("-------------------------------------")
    print(f"Expected Loss: {expected_loss:.4f}")

    # computing the worst case loss at the uncertainty level h
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
    x = np.arange(-1.7, 1.85, 0.15)
    y = np.arange(-1.7, 1.85, 0.15)
    xv, yv = np.meshgrid(x, y)
    theta = i_theta.theta(torch.stack([torch.from_numpy(xv), torch.from_numpy(yv)], dim=2)).detach().numpy()

    # drawing the contour plot of the loss function
    xloss = np.arange(-1.7, 1.71, 0.01)
    yloss = np.arange(-1.7, 1.71, 0.01)
    xlossv, ylossv = np.meshgrid(xloss, yloss)
    zloss = loss.cost(torch.from_numpy(xlossv), torch.from_numpy(ylossv))

    # # drawing the contour plot of the epicenter density
    # xepi = np.arange(-2.5, 2.5, 0.01)
    # yepi = np.arange(-2.5, 2.5, 0.01)
    # xepiv, yepiv = np.meshgrid(xepi, yepi)
    # zepi = mu.density(np.stack([xepiv, yepiv], axis=2))

    # Depict illustration
    fig, ax = plt.subplots()
    CS = ax.contour(xlossv, ylossv, zloss, cmap='viridis', levels=7)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.quiver(xv, yv, theta[:, :, 0], theta[:, :, 1], color='g')
    # plt.streamplot(xv, yv, theta[:, :, 0], theta[:, :, 1], density=1.4, linewidth=None, color='#A23BEC')
    ax.plot(epicenter[0], epicenter[1], 'rh')
    plt.title(f'Parametric optimizer for uncertainty level h={uncertainty_level}')
    plt.show()
