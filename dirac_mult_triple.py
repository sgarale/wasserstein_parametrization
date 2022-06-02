import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import cost_functions as cf


class Dirac2dMult:

    def __init__(self, atoms, weights=None):
        """
        :param atoms: list of list or numpy ndarray
        :param weights: list or numpy ndarray
        """
        self.atoms_x = np.array(atoms)[:, 0]
        self.atoms_y = np.array(atoms)[:, 1]
        if weights == None:
            self.weights = np.repeat([1./self.atoms_x.shape[0]], self.atoms_x.shape[0])
        else:
            self.weights = np.array(weights)
        if np.sum(self.weights) != 1.:
            raise Exception("Not a probability measure, check the weights.")

    def integrate(self, f, theta=np.array([[0,0]]), power=1):
        """
        Integrate against the weighted dirac measure.
        :param f: function
        :param theta: numpy ndarray
        :return: float
        """
        return np.sum(self.weights * np.power(f(self.atoms_x + theta[:, 0], self.atoms_y + theta[:, 1]), power))


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
    return - (distr.integrate(f=cost, theta=theta.reshape(-1, 2)) - h * penalty.evaluate(wass2_mult(distr, theta=theta.reshape(-1, 2)) / h))


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__=='__main__':

    # ---- Plots subfolder --------
    plot_fold = 'triple_epicenter'
    # -----------------------------
    plot_fold = f"plots/{plot_fold}"
    check_dir(plot_fold)


    city_center = [0., 0.]
    cost_level = 1.
    epicenter_1 = [0.75, 0.]
    epicenter_2 = [0., 1.]
    epicenter_3 = [-1.30, 0.]

    mu = Dirac2dMult([epicenter_1, epicenter_2, epicenter_3], [1./3, 1./3, 1./3])
    penalty = PolyPenalty(3)
    gaussian_cost = cf.GaussianKernelCost2D(x_0=city_center[0], y_0=city_center[1], level=cost_level, radius=1.5)

    # Contour plot of the cost function
    x = np.arange(-1.7, 1.71, 0.01)
    y = np.arange(-1.7, 1.71, 0.01)
    xv, yv = np.meshgrid(x, y)
    zv = gaussian_cost.cost(xv, yv)

    fig, ax = plt.subplots()
    CS = ax.contour(xv, yv, zv, cmap='viridis', levels=7)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.plot(epicenter_1[0], epicenter_1[1], 'r.')
    ax.plot(epicenter_2[0], epicenter_2[1], 'g.')
    ax.plot(epicenter_3[0], epicenter_3[1], 'b.')
    plt.savefig(f"{plot_fold}/loss_and_apriori.png")

    print(f"Expected value: {mu.integrate(gaussian_cost.cost) : .4f}")

    x0 = np.array([0, 0, 0, 0, 0, 0])

    h_levels = np.arange(0.03, 0.33, 0.03)
    I_theta = [mu.integrate(gaussian_cost.cost)]
    print("Directions of the minimization           loss")
    for h in h_levels:
        res = minimize(loss_mult, x0=x0, args=(h, mu, gaussian_cost.cost, penalty))
        ax.plot(epicenter_1[0] + res.x[0], epicenter_1[1] + res.x[1], 'r.')
        ax.plot(epicenter_2[0] + res.x[2], epicenter_2[1] + res.x[3], 'g.')
        ax.plot(epicenter_3[0] + res.x[4], epicenter_3[1] + res.x[5], 'b.')
        plt.savefig(f"{plot_fold}/optimizers_unc_{h:0.2f}.png")
        loss_tmp = - loss_mult(res.x, h, mu, gaussian_cost.cost, penalty)
        I_theta.append(loss_tmp)
        print(f"uncertainty level: {h : .2f}, directions of optimization: {res.x[0] : .4f}, {res.x[1] : .4f}, "
              f"{res.x[2] : .4f}, {res.x[3] : .4f}, {res.x[4] : .4f}, {res.x[5] : .4f}    loss: {loss_tmp: .6f}")
    plt.show()

    plt.clf()
    plt.plot(np.concatenate([[0],h_levels]), I_theta)
    plt.xlabel("Uncertainty level")
    plt.ylabel("Worst case loss")
    plt.savefig(f"{plot_fold}/worst_case_loss.png")
    plt.show()
