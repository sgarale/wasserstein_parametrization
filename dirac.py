import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import cost_functions as cf


class Dirac2d:

    def __init__(self, atoms):
        """
        :param atoms: list or numpy ndarray
        """
        self.atoms_x = np.array(atoms)[0]
        self.atoms_y = np.array(atoms)[1]

    def integrate(self, f, theta=np.array([0,0]), power=1):
        """
        Integrate against the weighted dirac measure.
        :param f: function
        :param theta: numpy ndarray
        :return: float
        """
        return np.power(f(self.atoms_x + theta[0], self.atoms_y + theta[1]), power)[0]



class Dirac2dMult:

    def __init__(self, atoms, weights=None):
        """
        :param atoms: list of list or numpy ndarray
        :param weights: list or numpy ndarray
        """
        self.atoms_x = np.array(atoms)[:, 0]
        self.atoms_y = np.array(atoms)[:, 1]
        if weights == None:
            weights = np.repeat([1./self.atoms_x.shape[0]], self.atoms_x.shape[0])
        else:
            self.weights = np.array(weights)
        if np.sum(weights) != 1.:
            raise Exception("Not a probability measure, check the weights.")

    def integrate(self, f, theta=np.array([[0,0]]), power=1):
        """
        Integrate against the weighted dirac measure.
        :param f: function
        :param theta: numpy ndarray
        :return: float
        """
        return np.sum(self.weights * np.power(f(self.atoms_x + theta[:, 0], self.atoms_y + theta[:, 1]), power))


def wass2(theta):
    """
    It computes the Wasserstein distance of order 2 between the measure mu and the measure mu_theta, where mu is a dirac measure.
    :param theta: numpy ndarray
    :return: float
    """
    return np.sqrt(np.power(theta, 2).sum())


def wass2_mult(dirac: Dirac2dMult, theta):
        """
        It computes the Wasserstein distance of order 2 between the measure mu and the measure mu_theta, where mu is a
        convex sum of dirac measure.
        :param dirac: Dirac2d
        :param theta: numpy ndarray
        :return: float
        """
        return np.sqrt((dirac.weights * np.power(theta, 2).sum(axis=1)).sum())


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

    def evaluate(self,x):
        if x < 0:
            raise Exception(f"Penalty function is defined only for positive numbers. Passed {x: .4f} instead.")
        return np.power(x, self.p)


def loss(theta, h, distr, cost, penalty: Penalty):
    """
    Compute the loss function for the optimization.
    :param h: float
    :param distr: Dirac2d
    :param cost: function of two variables
    :param penalty: Penalty
    :param theta: list or numpy ndarray (len = 2)
    :return:
    """
    return - (distr.integrate(f=cost, theta=theta) - h * penalty.evaluate(wass2(theta=theta) / h))


def loss_mult(theta, h, distr, cost, penalty: Penalty):
    """
    Compute the loss function for the optimization.
    :param h: float
    :param distr: Dirac2dMult
    :param cost: function of two variables
    :param penalty: Penalty
    :param theta: numpy ndarray (shape = d*2)
    :return:
    """
    return - (distr.integrate(f=cost, theta=theta) - h * penalty.evaluate(wass2_mult(distr, theta=theta) / h))



if __name__=='__main__':

    city_center = [0., 0.]
    cost_level = np.exp(1)
    epicenter = [0.5, 0.5]

    mu = Dirac2d(epicenter)
    penalty = PolyPenalty(3)

    def cost(x, y):
        return cf.gaussian_kernel_cost_2d(x, y, x_0=city_center[0], y_0=city_center[1], level=cost_level)

    print(f"Expected value: {mu.integrate(cost) : .4f}")

    x0 = np.array([0, 0])

    x = np.arange(0, 1.01, 0.01)
    h_levels = np.arange(0.01, 1.01, 0.01)
    I_theta = [mu.integrate(cost)]
    for h in h_levels:
        res = minimize(loss, x0=x0, args=(h, mu, cost, penalty))
        I_theta.append(- loss(res.x, h, mu, cost, penalty))

    plt.plot(x, I_theta)
    plt.show()


    exit()

    city_center_2 = [1.1, 1.1]
    def cost_2(x, y):
        cost = 0.5 * cf.gaussian_kernel_cost_2d(x, y, x_0=city_center[0], y_0=city_center[1], level=cost_level) \
               + 0.5 * cf.gaussian_kernel_cost_2d(x, y, x_0=city_center_2[0], y_0=city_center_2[1], level=cost_level)
        return cost

    # x_1 = np.arange(-1.5, 3., 0.01)
    # x_2 = np.arange(-1.5, 3., 0.01)
    # xv, yv = np.meshgrid(x_1, x_2)
    # zv = cost_2(xv, yv)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xv, yv, zv, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.set_title('Cost function')
    # plt.show()

    print(f"Expected value 2: {mu.integrate(cost_2) : .4f}")

    x = np.arange(0, 1.01, 0.01)
    h_levels = np.arange(0.01, 1.01, 0.01)
    I_theta = [mu.integrate(cost_2)]
    for h in h_levels:
        res = minimize(loss, x0=x0, args=(h, mu, cost_2, penalty))
        I_theta.append(- loss(res.x, h, mu, cost_2, penalty))

    plt.plot(x, I_theta)
    plt.show()

