import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import utils as ut

class GaussianKernelCost2D:
    """
    Differentiable cost function with compact support and gaussian kernel shape.
    """

    def __init__(self, x_0, y_0, level=1., radius=1.):
        """
        :param x_0: float
        :param y_0: float
        :param level: float
        :param radius: float
        """
        self.x0 = float(x_0)
        self.y0 = float(y_0)
        self.level = float(level)
        self.radius = float(radius)

    def cost(self, x, y):
        """
        :param x: flaot or numpy ndarray
        :param y: flaot or numpy ndarray
        :return: numpy ndarray
        """
        if isinstance(x, float):
            x = np.array([x])
            y = np.array([y])

        distance_2 = np.power(x - self.x0, 2) + np.power(y - self.y0, 2)
        cost = np.zeros(x.shape)
        mask = distance_2 < np.power(self.radius, 2)
        cost[mask] = self.level * np.exp(1. + np.power(self.radius, 2) / (distance_2[mask] - np.power(self.radius, 2)))
        return cost

    def gradient(self, x, y):
        """
        Gives the gradient of the function at (x, y).
        :param x: float or numpy ndarray
        :param y: float or numpy ndarray
        :return: list of numpy ndarray
        """
        if isinstance(x, float):
            x = np.array([x])
            y = np.array([y])

        cost = self.cost(x, y)
        mask = cost > 0
        denominator = np.power(x - self.x0, 2) + np.power(y - self.y0, 2) - self.radius * self.radius
        nablax = np.zeros(x.shape)
        nablax[mask] = - 2. * cost[mask] * (x[mask] - self.x0) * np.power(self.radius / denominator[mask], 2)
        nablay = np.zeros(y.shape)
        nablay[mask] = - 2. * cost[mask] * (y[mask] - self.y0) * np.power(self.radius / denominator[mask], 2)
        return [nablax, nablay]



class GaussianKernelCost2DTensor:
    """
    Pytorch implementation of differentiable cost function with compact support and gaussian kernel shape.
    """

    def __init__(self, x_0, y_0, level=1., radius=1.):
        """
        :param x_0: float
        :param y_0: float
        :param level: float
        :param radius: float
        """
        self.x0 = torch.tensor(x_0)
        self.y0 = torch.tensor(y_0)
        self.level = torch.tensor(level)
        self.radius = torch.tensor(radius)

    def cost(self, x, y):
        """
        :param x: flaot or torch tensor
        :param y: flaot or torch tensor
        :return: pytorch tensor
        """
        if isinstance(x, float):
            x = torch.tensor([x])
            y = torch.tensor([y])

        distance_2 = torch.pow(x - self.x0, 2) + torch.pow(y - self.y0, 2)
        cost = torch.zeros(x.shape)
        mask = distance_2 < torch.pow(self.radius, 2)
        cost[mask] = self.level * torch.exp(1. + torch.pow(self.radius, 2) / (torch.masked_select(distance_2, mask) - torch.pow(self.radius, 2)))
        return cost

    def gradient(self, x, y):
        """
        Gives the gradient of the function at the points x, y.
        :param x: float or pytorch tensor
        :param y: float or pytorch tensor
        :return: list of pytorch tensors
        """
        if isinstance(x, float):
            x = torch.tensor([x])
            y = torch.tensor([y])

        denominator = torch.pow(x - self.x0, 2) + torch.pow(y - self.y0, 2) - self.radius * self.radius
        cost = self.cost(x, y)
        mask = cost > 0
        nablax = torch.zeros(x.shape)
        nablax[mask] = - 2. * cost[mask] * (x[mask] -self.x0) * torch.pow(self.radius / denominator[mask], 2)
        nablay = torch.zeros(x.shape)
        nablay[mask] = - 2. * cost[mask] * (y[mask] - self.y0) * torch.pow(self.radius / denominator[mask], 2)
        return [nablax, nablay]



class DoubleGaussianKernelCost2DTensor:
    """
    Differentiable cost function with compact support and double dome shape.
    """

    def __init__(self, x_1, y_1, x_2, y_2, level_1=1., radius_1=1., level_2=1., radius_2=1.):
        """
        :param x_1: float
        :param y_1: float
        :param x_2: float
        :param y_2: float
        :param level_1: float
        :param radius_1: float
        :param level_2: float
        :param radius_2: float
        """
        self.dome_1 = GaussianKernelCost2DTensor(x_0=x_1, y_0=y_1, level=level_1, radius=radius_1)
        self.dome_2 = GaussianKernelCost2DTensor(x_0=x_2, y_0=y_2, level=level_2, radius=radius_2)

    def cost(self, x, y):
        """
        :param x: flaot or numpy ndarray or torch tensor
        :param y: flaot or numpy ndarray or torch tensor
        :return: pytorch tensor
        """
        return self.dome_1.cost(x, y) + self.dome_2.cost(x, y)

    def gradient(self, x, y):
        """
        Gives the gradient of the function at the points x, y.
        :param x: float or numpy ndarray or pytorch tensor
        :param y: float or numpy ndarray or pytorch tensor
        :return: list of pytorch tensor
        """
        nabla_1 = self.dome_1.gradient(x, y)
        nabla_2 = self.dome_2.gradient(x, y)
        return [nabla_1[0] + nabla_2[0], nabla_1[1] + nabla_2[1]]



class BullSpread:
    """
    Payoff of the bull call spread option with lower strike K_long and upper strike K_short.
    Payoff: (x - K_long)^+ - (x - K_short)^+
    """

    def __init__(self, K_long, K_short):
        """
        :param K_long: float
        :param K_short: float
        """
        self.K_long = torch.tensor(K_long)
        self.K_short = torch.tensor(K_short)

    def cost(self, x):
        """
        :param x: float or pytorch tensor
        :return: pytorch tensor
        """
        return torch.maximum(x - self.K_long, torch.tensor(0)) - torch.maximum(x - self.K_short, torch.tensor(0))



if __name__=='__main__':

    # This main plots the cost function for the earthquake model of Section 4.

    # setting latex style for plots
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 13
    plt.rcParams['legend.fontsize'] = 13

    # setting the output folder
    ut.check_dir("plots")

    # -------- insert center points, level, and radius -----------
    x_0 = 0.
    y_0 = 0.
    level = 1.
    radius = 1.5
    gauss_cost = GaussianKernelCost2D(x_0, y_0, level, radius)
    x_1 = np.arange(-1.8, 1.8, 0.01)
    x_2 = np.arange(-1.8, 1.8, 0.01)
    xv, yv = np.meshgrid(x_1, x_2)
    zg = gauss_cost.cost(xv, yv)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1, 1, 0.4))
    ax.plot_surface(xv, yv, zg, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    plt.tight_layout()
    plt.savefig(os.path.join('plots', "loss_plot.eps"), format='eps')
    plt.show()

    # Depict the gradient of the gaussian kernel loss
    x_1 = np.arange(-1.2, 1.2, 0.01)
    x_2 = np.arange(-1.2, 1.2, 0.01)
    xv, yv = np.meshgrid(x_1, x_2)
    zg = gauss_cost.cost(xv, yv)

    x_1 = np.arange(-1.6, 1.7, 0.10)
    x_2 = np.arange(-1.6, 1.7, 0.10)
    xvgrad, yvgrad = np.meshgrid(x_1, x_2)
    zgrad = gauss_cost.gradient(xvgrad, yvgrad)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    CS = ax.contour(xv, yv, np.array(zg), cmap='viridis', levels=7)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.quiver(xvgrad, yvgrad, np.array(zgrad[0]), np.array(zgrad[1]), color='g')
    plt.title(f'Gradient of the gaussian kernel loss')
    plt.show()