import sys
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

# Add the parent folder to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import cost_functions as cf
import penalties
import utils as ut

class Dirac2dMult:
    """
    Class implementing a measure given by a convex sum of Dirac measures
    """

    def __init__(self, atoms, weights=None):
        """
        :param atoms: list of lists or numpy ndarrays
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

    def integrate(self, f, theta=np.array([[0, 0]]), power=1):
        """
        Integrate under the measure mu_theta.
        :param f: function
        :param theta: numpy ndarray
        :return: float
        """
        return np.sum(self.weights * np.power(f(self.atoms_x + theta[:, 0], self.atoms_y + theta[:, 1]), power))


def wass2_mult(dirac: Dirac2dMult, theta):
    """
    Computes the Wasserstein distance of order 2 between the measure mu and the measure mu_theta, where mu is a
    convex sum of Dirac measures.
    :param dirac: Dirac2dMult
    :param theta: numpy ndarray
    :return: float
    """
    return np.sqrt((dirac.weights * np.power(theta, 2).sum(axis=1)).sum())


def loss_mult(theta, h, distr, cost, penalty: penalties.Penalty):
    """
    Compute the loss function for the optimization.
    :param h: float
    :param distr: Dirac2dMult
    :param cost: function of two variables
    :param penalty: Penalty
    :param theta: numpy ndarray (shape = d*2)
    :return: float
    """
    return - (distr.integrate(f=cost, theta=theta.reshape(-1, 2)) - h * penalty.evaluate(wass2_mult(distr, theta=theta.reshape(-1, 2)) / h))



if __name__=='__main__':

    # ---- Plots subfolder --------
    plot_fold = 'triple_epicenter'
    # ---------- INPUTS -----------------
    city_center = [0., 0.]
    cost_level = 1.
    radius = 1.5
    epicenter_1 = [0.55, 0.]
    epicenter_2 = [0., .85]
    epicenter_3 = [-1.10, 0.]
    weights = [1./3, 1./3, 1./3]
    h_levels = np.arange(0.015, 0.315, 0.015)
    # -----------------------------------

    # setting latex style for plots
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 13
    plt.rcParams['legend.fontsize'] = 13

    # checking the existence of the plot folder
    plot_fold = os.path.join('plots', plot_fold)
    ut.check_dir(plot_fold)

    # Initializing measure, penalty, and loss function
    mu = Dirac2dMult([epicenter_1, epicenter_2, epicenter_3], weights)
    penalty = penalties.PolyPenalty(2)
    gaussian_cost = cf.GaussianKernelCost2D(x_0=city_center[0], y_0=city_center[1], level=cost_level, radius=radius)

    # Contour plot of the cost function
    x = np.arange(-1.7, 1.71, 0.01)
    y = np.arange(-1.7, 1.71, 0.01)
    xv, yv = np.meshgrid(x, y)
    zv = gaussian_cost.cost(xv, yv)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    CS = ax.contour(xv, yv, zv, cmap='viridis', levels=7)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.plot(epicenter_1[0], epicenter_1[1], 'r.')
    ax.plot(epicenter_2[0], epicenter_2[1], 'g.')
    ax.plot(epicenter_3[0], epicenter_3[1], 'b.')
    plt.savefig(os.path.join(plot_fold, 'loss_and_apriori.png'), bbox_inches='tight')
    plt.savefig(os.path.join(plot_fold, 'loss_and_apriori.eps'), format='eps', bbox_inches='tight')

    print(f"Expected loss: {mu.integrate(gaussian_cost.cost) :.4f}")


    # Optimization cycle
    x0 = np.array([0, 0, 0, 0, 0, 0]) # initial guess for the optimization

    I_theta = [mu.integrate(gaussian_cost.cost)]
    print("Directions of the minimization           loss")
    for h in h_levels:
        res = minimize(loss_mult, x0=x0, args=(h, mu, gaussian_cost.cost, penalty))
        ax.plot(epicenter_1[0] + res.x[0], epicenter_1[1] + res.x[1], 'r.')
        ax.plot(epicenter_2[0] + res.x[2], epicenter_2[1] + res.x[3], 'g.')
        ax.plot(epicenter_3[0] + res.x[4], epicenter_3[1] + res.x[5], 'b.')
        plt.savefig(os.path.join(plot_fold, f'optimizers_unc_{h:0.2f}.png'), bbox_inches='tight')
        plt.savefig(os.path.join(plot_fold, f'optimizers_unc_{h:0.2f}.eps'), format='eps', box_inches='tight')
        loss_tmp = - loss_mult(res.x, h, mu, gaussian_cost.cost, penalty)
        I_theta.append(loss_tmp)
        print(f"uncertainty level: {h : .2f}, directions of optimization: {res.x[0] : .4f}, {res.x[1] : .4f}, "
              f"{res.x[2] : .4f}, {res.x[3] : .4f}, {res.x[4] : .4f}, {res.x[5] : .4f}    loss: {loss_tmp: .6f}")
    plt.show()

    plt.clf()
    plt.plot(np.concatenate([[0],h_levels]), I_theta)
    plt.xlabel("Uncertainty level")
    plt.ylabel("Worst case loss")
    # plt.savefig(os.path.join(plot_fold, f'worst_case_loss.png'), bbox_inches='tight')
    plt.savefig(os.path.join(plot_fold, f'worst_case_loss.eps'), format='eps', bbox_inches='tight')
    plt.show()
