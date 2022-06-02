import numpy as np
import matplotlib.pyplot as plt
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
    cost[mask] = level * np.exp(- 1. / (radius - np.power(distance[mask], 2)))
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
        cost[mask] = self.level * np.exp(- 1. / (np.power(self.radius, 2) - distance_2[mask]))
        return cost


if __name__=='__main__':

    # one dimensional case
    x_0 = 2
    x = np.arange(0, 4, 0.01)

    # distance cost function
    y_d = [distance_cost(i, x_0) for i in x]
    y_g = [gaussian_kernel_cost(i, x_0) for i in x]
    plt.plot(x, y_d)
    plt.plot(x, y_g)
    plt.legend(["Distance cost", "Gaussian kernel cost"])
    plt.title("Cost functions")
    plt.show()

    # two dimensional case
    # -------- insert center points, level, and radius -----------
    x_0 = 2.
    y_0 = 2.
    level = 1.
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