import math

import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D



def gaussian_kernel_cost(x, x_0, level=1., radius=1.):
    """
    Differentiable cost function with compact support and gaussian kernel shape.
    :param x: float or numpy ndarray
    :param x_0: float or numpy ndarray
    :return: float
    """
    if isinstance(x, float):
        x = np.array([x])
    cost = np.zeros(x.shape)
    distance = np.absolute(x - x_0)
    mask = distance < radius
    cost[mask] = level * np.exp(1. - radius / (radius - np.power(distance[mask], 2)))
    return cost


def distance_cost(x, x_0, level=1., radius=1.):
    """
    Distance cost function with compact support
    :param x:
    :param x_0:
    :param radius:
    :return:
    """
    if isinstance(x, float):
        x = np.array([x])

    cost = np.zeros(x.shape)
    distance = np.absolute(x - x_0)
    mask = distance < radius
    cost[mask] = (level - (level / radius) * distance[mask])
    return cost


def distance_cost_2d(x, y, x_0, y_0, level=1., radius=1.):
    """
    Distance cost function with compact support
    :param x: float or numpy ndarray
    :param y: float or numpy ndarray
    :param x_0: float or numpy ndarray
    :param y_0: float or numpy ndarray
    :param level: float
    :param radius: float
    :return: float or numpy ndarray
    """
    if isinstance(x, float):
        x = np.array([x])
        y = np.array([y])

    distance = np.sqrt(np.power(x - x_0, 2) + np.power(y - y_0, 2))
    cost = np.zeros(x.shape)
    mask = distance < radius
    cost[mask] = (level - (level / radius) * distance[mask])
    return cost


def gaussian_kernel_cost_2d(x, y, x_0, y_0, level=1., radius=1.):
    """
    Differentiable cost function with compact support and gaussian kernel shape.
    :param x: float or numpy ndarray
    :param y: float or numpy ndarray
    :param x_0: float or numpy ndarray
    :param y_0:float or numpy ndarray
    :return: float or numpy ndarray
    """
    if isinstance(x, float):
        x = np.array([x])
        y = np.array([y])

    distance_2 = np.power(x - x_0, 2) + np.power(y - y_0, 2)
    cost = np.zeros(x.shape)
    mask = distance_2 < radius * radius
    cost[mask] = level * np.exp(- 1. / (radius * radius - distance_2[mask]))
    return cost


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
        Gives the gradient of the function at the points x, y.
        :param x: float or numpy ndarray
        :param y: float or numpy ndarray
        :return: list of numpy ndarray
        """
        if isinstance(x, float):
            x = np.array([x])
            y = np.array([y])

        denominator = np.power(x - self.x0, 2) + np.power(y - self.y0, 2) - self.radius * self.radius
        cost = self.cost(x, y)
        nablax = - 2. * cost * (x - self.x0) * np.power(self.radius / denominator, 2)
        nablay = - 2. * cost * (y - self.y0) * np.power(self.radius / denominator, 2)
        return [nablax, nablay]



class GaussianKernelCost2DTensor:
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
        :return: list of pytorch tensor
        """
        if isinstance(x, float):
            x = torch.tensor([x])
            y = torch.tensor([y])

        denominator = torch.pow(x - self.x0, 2) + torch.pow(y - self.y0, 2) - self.radius * self.radius
        cost = self.cost(x, y)
        nablax = - 2. * cost * (x -self.x0) * torch.pow(self.radius / denominator, 2)
        nablay = - 2. * cost * (y - self.y0) * torch.pow(self.radius / denominator, 2)
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
        :param x: flaot or torch tensor
        :param y: flaot or torch tensor
        :return: pytorch tensor
        """
        return self.dome_1.cost(x, y) + self.dome_2.cost(x, y)

    def gradient(self, x, y):
        """
        Gives the gradient of the function at the points x, y.
        :param x: float or pytorch tensor
        :param y: float or pytorch tensor
        :return: list of pytorch tensor
        """
        nabla_1 = self.dome_1.gradient(x, y)
        nabla_2 = self.dome_2.gradient(x, y)
        return [nabla_1[0] + nabla_2[0], nabla_1[1] + nabla_2[1]]


if __name__=='__main__':

    # This main tests the different cost functions and the different math kernels used for the
    # computations (numpy and pytorch)

    # # one dimensional case
    # x_0 = 2
    # x = np.arange(0, 4, 0.01)
    #
    # # distance cost function
    # y_d = [distance_cost(i, x_0) for i in x]
    # y_g = [gaussian_kernel_cost(i, x_0, level=0.5) for i in x]
    # plt.plot(x, y_d)
    # plt.plot(x, y_g)
    # plt.legend(["Distance cost", "Gaussian kernel cost"])
    # plt.title("Cost functions")
    # plt.show()
    #
    # # Figure xx.xx
    # # -------- insert center points, level, and radius -----------
    # x_0 = 0.
    # y_0 = 0.
    # level = 1.
    # radius = 1.5
    # gauss_cost = GaussianKernelCost2D(x_0, y_0, level, radius)
    # x_1 = np.arange(-1.8, 1.8, 0.01)
    # x_2 = np.arange(-1.8, 1.8, 0.01)
    # xv, yv = np.meshgrid(x_1, x_2)
    # zg = gauss_cost.cost(xv, yv)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect((1,1,0.4))
    # ax.plot_surface(xv, yv, zg, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    # plt.show()
    # # exit()
    #
    # # two dimensional case with numpy functions
    # # -------- insert center points, level, and radius -----------
    # x_0 = 2.
    # y_0 = 2.
    # level = 0.5
    # radius = 1.
    # gauss_cost = GaussianKernelCost2D(x_0, y_0, level, radius)
    # x_1 = np.arange(0, 4, 0.01)
    # x_2 = np.arange(0, 4, 0.01)
    # xv, yv = np.meshgrid(x_1, x_2)
    # zd = distance_cost_2d(xv, yv, 2, 2)
    # zg = gauss_cost.cost(xv, yv)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xv, yv, zd, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.plot_surface(xv, yv, zg, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # # ax.legend(["Distance cost", "Gaussian kernel cost"])
    # ax.set_title('2D cost functions')
    # plt.show()
    #
    #
    # # two dimensional case with pytorch functions
    # # -------- insert center points, level, and radius -----------
    # torch.set_default_dtype(torch.float64)
    # x_0 = 0.
    # y_0 = 0.
    # level = 0.5
    # radius = 1.
    # gauss_cost = GaussianKernelCost2DTensor(x_0, y_0, level, radius)
    # x_1 = np.arange(-2, 2, 0.01)
    # x_2 = np.arange(-2, 2, 0.01)
    # xv, yv = np.meshgrid(x_1, x_2)
    # zd = distance_cost_2d(xv, yv, 0, 0)
    # zg = gauss_cost.cost(torch.from_numpy(xv), torch.from_numpy(yv))
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xv, yv, zd, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.plot_surface(xv, yv, np.array(zg), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # # ax.legend(["Distance cost", "Gaussian kernel cost"])
    # ax.set_title('2D cost functions (pytorch math functions)')
    # plt.show()

    # Depict gradient of the gaussian kernel loss
    # x_0 = 0.
    # y_0 = 0.
    # level = 0.5
    # radius = 1.
    # gauss_cost = GaussianKernelCost2D(x_0, y_0, level, radius)
    # x_1 = np.arange(-1.2, 1.2, 0.01)
    # x_2 = np.arange(-1.2, 1.2, 0.01)
    # xv, yv = np.meshgrid(x_1, x_2)
    # zg = gauss_cost.cost(xv, yv)
    #
    # x_1 = np.arange(-1.2, 1.3, 0.10)
    # x_2 = np.arange(-1.2, 1.3, 0.10)
    # xvgrad, yvgrad = np.meshgrid(x_1, x_2)
    # zgrad = gauss_cost.gradient(xvgrad, yvgrad)
    #
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # CS = ax.contour(xv, yv, np.array(zg), cmap='viridis', levels=7)
    # ax.clabel(CS, inline=True, fontsize=10)
    # ax.quiver(xvgrad, yvgrad, np.array(zgrad[0]), np.array(zgrad[1]), color='g')
    # plt.title(f'Gradient of the gaussian kernel loss')
    # plt.show()

    # # Depict gradient of the gaussian kernel loss using pytorch functions
    # torch.set_default_dtype(torch.float64)
    # x_0 = 0.
    # y_0 = 0.
    # level = 0.5
    # radius = 1.
    # gauss_cost = GaussianKernelCost2DTensor(x_0, y_0, level, radius)
    # x_1 = np.arange(-1.2, 1.2, 0.01)
    # x_2 = np.arange(-1.2, 1.2, 0.01)
    # xv, yv = np.meshgrid(x_1, x_2)
    # zg = gauss_cost.cost(torch.from_numpy(xv), torch.from_numpy(yv))
    #
    # x_1 = np.arange(-1.2, 1.3, 0.1)
    # x_2 = np.arange(-1.2, 1.3, 0.1)
    # xvgrad, yvgrad = np.meshgrid(x_1, x_2)
    # zgradt = gauss_cost.gradient(torch.from_numpy(xvgrad), torch.from_numpy(yvgrad))
    #
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # CS = ax.contour(xv, yv, np.array(zg), cmap='viridis', levels=7)
    # ax.clabel(CS, inline=True, fontsize=10)
    # ax.quiver(xvgrad, yvgrad, np.array(zgradt[0]), np.array(zgradt[1]), color='g')
    # plt.title(f'Gradient of the gaussian kernel loss (pytorch functions)')
    # plt.show()
    #
    # print(f"Nabla x is equal: {np.allclose(zgrad[0], np.array(zgradt[0]))}")
    # print(f"Nabla y is equal: {np.allclose(zgrad[1], np.array(zgradt[1]))}")

    # double dome cost with pytorch functions
    # -------- insert center points, level, and radius -----------
    torch.set_default_dtype(torch.float64)
    x_1 = 0.
    y_1 = 0.
    level_1 = 0.5
    radius_1 = 1.
    x_2 = 1.25
    y_2 = 0.
    level_2 = 0.3
    radius_2 = 0.75
    double_dome_cost = DoubleGaussianKernelCost2DTensor(x_1, y_1, x_2, y_2, level_1, radius_1, level_2, radius_2)
    x_ticks = np.arange(-1.2, 2.2, 0.01)
    y_ticks = np.arange(-1.2, 1.2, 0.01)
    xv, yv = np.meshgrid(x_ticks, y_ticks)
    zg = double_dome_cost.cost(torch.from_numpy(xv), torch.from_numpy(yv))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xv, yv, np.array(zg), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Double dome cost function')
    plt.show()

    # Depict gradient of the double dome loss using pytorch functions
    torch.set_default_dtype(torch.float64)
    x_1 = 0.
    y_1 = 0.
    level_1 = 0.5
    radius_1 = 1.
    x_2 = 1.25
    y_2 = 0.
    level_2 = 0.3
    radius_2 = 0.75
    double_dome_cost = DoubleGaussianKernelCost2DTensor(x_1, y_1, x_2, y_2, level_1, radius_1, level_2, radius_2)
    x_ticks = np.arange(-1.2, 2.2, 0.01)
    y_ticks = np.arange(-1.2, 1.2, 0.01)
    xv, yv = np.meshgrid(x_ticks, y_ticks)
    zg = double_dome_cost.cost(torch.from_numpy(xv), torch.from_numpy(yv))

    x_ticks_coarse = np.arange(-1.2, 2.2, 0.1)
    y_ticks_coarse = np.arange(-1.2, 1.3, 0.1)
    xvgrad, yvgrad = np.meshgrid(x_ticks_coarse, y_ticks_coarse)
    zgradt = double_dome_cost.gradient(torch.from_numpy(xvgrad), torch.from_numpy(yvgrad))

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    CS = ax.contour(xv, yv, np.array(zg), cmap='viridis', levels=7)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.quiver(xvgrad, yvgrad, np.array(zgradt[0]), np.array(zgradt[1]), color='g')
    plt.title(f'Gradient of the gaussian kernel loss (pytorch functions)')
    plt.show()