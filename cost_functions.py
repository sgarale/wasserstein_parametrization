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
        nablax = - 2. * cost * x * np.power(self.radius / denominator, 2)
        nablay = - 2. * cost * y * np.power(self.radius / denominator, 2)
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
        :return: numpy ndarray
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
        nablax = - 2. * cost * x * torch.pow(self.radius / denominator, 2)
        nablay = - 2. * cost * y * torch.pow(self.radius / denominator, 2)
        return [nablax, nablay]



if __name__=='__main__':

    # This main tests the different cost functions and the different math kernels used for the
    # computations (numpy and pytorch)

    # one dimensional case
    x_0 = 2
    x = np.arange(0, 4, 0.01)

    # distance cost function
    y_d = [distance_cost(i, x_0) for i in x]
    y_g = [gaussian_kernel_cost(i, x_0, level=0.5) for i in x]
    plt.plot(x, y_d)
    plt.plot(x, y_g)
    plt.legend(["Distance cost", "Gaussian kernel cost"])
    plt.title("Cost functions")
    plt.show()

    # Figure xx.xx
    x_0 = 0.
    y_0 = 0.
    level = 0.5
    radius = 1.
    gauss_cost = GaussianKernelCost2D(x_0, y_0, level, radius)
    x_1 = np.arange(-1.5, 1.5, 0.01)
    x_2 = np.arange(-1.5, 1.5, 0.01)
    xv, yv = np.meshgrid(x_1, x_2)
    zg = gauss_cost.cost(xv, yv)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xv, yv, zg, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    plt.show()
    exit()

    # two dimensional case with numpy functions
    # -------- insert center points, level, and radius -----------
    x_0 = 2.
    y_0 = 2.
    level = 0.5
    radius = 1.
    gauss_cost = GaussianKernelCost2D(x_0, y_0, level, radius)
    x_1 = np.arange(0, 4, 0.01)
    x_2 = np.arange(0, 4, 0.01)
    xv, yv = np.meshgrid(x_1, x_2)
    zd = distance_cost_2d(xv, yv, 2, 2)
    zg = gauss_cost.cost(xv, yv)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xv, yv, zd, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.plot_surface(xv, yv, zg, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.legend(["Distance cost", "Gaussian kernel cost"])
    ax.set_title('2D cost functions')
    plt.show()


    # two dimensional case with pytorch functions
    # -------- insert center points, level, and radius -----------
    torch.set_default_dtype(torch.float64)
    x_0 = 0.
    y_0 = 0.
    level = 0.5
    radius = 1.
    gauss_cost = GaussianKernelCost2DTensor(x_0, y_0, level, radius)
    x_1 = np.arange(-2, 2, 0.01)
    x_2 = np.arange(-2, 2, 0.01)
    xv, yv = np.meshgrid(x_1, x_2)
    zd = distance_cost_2d(xv, yv, 0, 0)
    zg = gauss_cost.cost(torch.from_numpy(xv), torch.from_numpy(yv))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xv, yv, zd, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.plot_surface(xv, yv, np.array(zg), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.legend(["Distance cost", "Gaussian kernel cost"])
    ax.set_title('2D cost functions (pytorch math functions)')
    plt.show()

    # Depict gradient of the gaussian kernel loss
    x_0 = 0.
    y_0 = 0.
    level = 0.5
    radius = 1.
    gauss_cost = GaussianKernelCost2D(x_0, y_0, level, radius)
    x_1 = np.arange(-1.2, 1.2, 0.01)
    x_2 = np.arange(-1.2, 1.2, 0.01)
    xv, yv = np.meshgrid(x_1, x_2)
    zg = gauss_cost.cost(xv, yv)

    x_1 = np.arange(-1.2, 1.2, 0.05)
    x_2 = np.arange(-1.2, 1.2, 0.05)
    xv2, yv2 = np.meshgrid(x_1, x_2)
    zgrad = gauss_cost.gradient(xv2, yv2)

    fig, ax = plt.subplots()
    CS = ax.contour(xv, yv, np.array(zg), cmap='viridis', levels=7)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.quiver(xv2, yv2, np.array(zgrad[0]), np.array(zgrad[1]), color='g')
    plt.title(f'Gradient of the gaussian kernel loss')
    plt.show()


    # Depict gradient of the gaussian kernel loss using pytorch functions
    torch.set_default_dtype(torch.float64)
    x_0 = 0.
    y_0 = 0.
    level = 0.5
    radius = 1.
    gauss_cost = GaussianKernelCost2DTensor(x_0, y_0, level, radius)
    x_1 = np.arange(-1.2, 1.2, 0.01)
    x_2 = np.arange(-1.2, 1.2, 0.01)
    xv, yv = np.meshgrid(x_1, x_2)
    zg = gauss_cost.cost(torch.from_numpy(xv), torch.from_numpy(yv))

    x_1 = np.arange(-1.2, 1.2, 0.05)
    x_2 = np.arange(-1.2, 1.2, 0.05)
    xv2, yv2 = np.meshgrid(x_1, x_2)
    zgradt = gauss_cost.gradient(torch.from_numpy(xv2), torch.from_numpy(yv2))

    fig, ax = plt.subplots()
    CS = ax.contour(xv, yv, np.array(zg), cmap='viridis', levels=7)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.quiver(xv2, yv2, np.array(zgradt[0]), np.array(zgradt[1]), color='g')
    plt.title(f'Gradient of the gaussian kernel loss (pytorch functions)')
    plt.show()

    print(f"Nabla x is equal: {np.allclose(zgrad[0], np.array(zgradt[0]))}")
    print(f"Nabla y is equal: {np.allclose(zgrad[1], np.array(zgradt[1]))}")