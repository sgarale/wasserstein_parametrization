import os
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
from torch import nn
import cost_functions as cf
import penalties as pnl
import gaussian_full as gauss


def blackscholes_option(S, K, T, r, q, sigma, flag='Call'):
    """
    Computes the analytical price of call and put option using the GBM model (Black-Scholes formula).
    :param S: float, initial level of the asset
    :param K: float, strike level
    :param T: float, maturity
    :param r: float, zero coupon risk free rate
    :param q: float, dividend rate
    :param sigma: float, volatility
    :param flag: str, available choices: "Call", "Put"
    :return: float
    """
    if flag == 'Call':
        w = 1
    elif flag == 'Put':
        w = -1
    else:
        raise Exception('Unknown option type.')

    d1 = (np.log(S / K) + (r - q + 0.5 * np.power(sigma, 2)) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = w * (np.exp(-q * T) * S * norm.cdf(w * d1) - np.exp(-r * T) * K * norm.cdf(w * d2))

    return price


class MartReLUNetwork(nn.Module):

    def __init__(self, width, depth=1):
        super(MartReLUNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, width)
        )
        for i in range(depth - 1):
            self.linear_relu_stack.append(nn.ReLU())
            self.linear_relu_stack.append(nn.Linear(width, width))
        self.linear_relu_stack.append(nn.ReLU())
        self.linear_relu_stack.append(nn.Linear(width, 1))

    def forward(self, x):
        theta = self.linear_relu_stack(x)
        return theta


class PenalisedLossMart(nn.Module):

    def __init__(self, func, penalty, p, h):
        super(PenalisedLossMart, self).__init__()
        self.func = func
        self.penalty = penalty
        self.p = p
        self.cp = torch.tensor(math.sqrt(2) * (math.gamma((p + 1) / 2.) / math.sqrt(math.pi)) ** (1. / p))
        self.randomizer = torch.distributions.normal.Normal(0., 1.)
        self.h = h

    def forward(self, y, theta_y):
        s = self.randomizer.sample(y.shape[0])
        integral = torch.mean(self.func(torch.exp(y + s * theta_y)))
        Lp_norm_theta = torch.pow(torch.mean(torch.pow(torch.abs(theta_y), self.p)), 1./self.p) #TODO: insert the log scaling
        penal_term = self.h * self.penalty(torch.pow(self.cp * Lp_norm_theta, 2) / self.h)
        return - (integral - penal_term)



class IThetaMart(nn.Module):

    def __init__(self, func, penalty, p, mu, h, width, depth, nr_sample):
        super(IThetaMart, self).__init__()
        self.theta = MartReLUNetwork(width, depth)
        self.mult_coeff = nn.Parameter(torch.tensor(.1))
        self.penalised_loss = PenalisedLossMart(func=func, penalty=penalty, p=p, h=h)
        self.mu = mu
        self.nr_sample = nr_sample

    def forward(self):
        y = self.mu.sample([self.nr_sample])
        theta_y = self.mult_coeff * self.theta(y)
        i_theta = self.penalised_loss(y, theta_y)
        return i_theta


if __name__ == '__main__':

    torch.set_default_dtype(torch.float64)
    # ---------------------- Inputs ----------------------
    plot_fold = 'call_option_0.08'
    eval_type = 'neural_network'  # choose among ['neural_network','one_dimensional','both']
    p = 3
    drift = 0.
    volatility = 0.20
    strike = 1.
    maturity = 0.08
    net_width = 20
    net_depth = 4
    mc_samples = 2**15
    learning_rate = 0.001
    epochs = 1100
    rolling_window = 100
    # ----------------------------------------------------

    plot_fold = f"plots/{plot_fold}"
    gauss.check_dir(plot_fold)

    # writing summary
    dump_summary = open(os.path.join(plot_fold, "summary.txt"), 'w')
    dump_summary.write(f'\norder: {p}, strike level: {strike}, drift: {drift}, volatility: {volatility}'
                       f'\nmaturity: {maturity}')
    dump_summary.write(f'\nneural network depth: {net_depth}\nneural network width: {net_width}'
                       f'\nmc saples: {mc_samples}\nlearning rate: {learning_rate}\nepochs: {epochs}'
                       f'\nrolling window: {rolling_window}')

    # fixing the seed
    torch.manual_seed(29)
    np.random.seed(3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    start = time.time()

    # initialization of loss function, baseline model, and penalisation function
    loss = cf.CallPayoff(strike)
    mu = torch.distributions.normal.Normal(drift, volatility)
    penalty = pnl.PolyPenaltyTensor(3) #TODO: change the penalty

    # computing the option fair value via Black&Scholes formula
    fv = blackscholes_option(S=1., K=strike, T=maturity, r=0., q=0., sigma=volatility)

    # computing the upper fair value bound at the uncertainty level h via nn optimization

    print("starting neural network optimization...")
    #initializing the neural network
    i_theta_mart = IThetaMart(func=loss.cost, penalty=penalty.evaluate, p=p, mu=mu,
                     h=maturity, width=net_width, depth=net_depth,
                     nr_sample=mc_samples)
    optm = torch.optim.Adam(i_theta_mart.parameters(), lr=learning_rate)
    out_vector = []
    for i in range(epochs):
        out = gauss.train(i_theta_mart, optm)
        out_vector.append(-float(out))
    out_vector = pd.Series(out_vector).rolling(rolling_window).mean()

    print(f"Parametrized worst case loss: {out_vector[out_vector.shape[0]-1]:.6f}")
    dump_summary.write(f'\nWorst case loss through neural network optimizer: {out_vector[out_vector.shape[0]-1]:.8f}')

    # plotting the training phase
    plt.plot(np.arange(1, out_vector.shape[0] + 1), out_vector, label='neural network upper model')
    plt.plot(np.arange(1, out_vector.shape[0] + 1), np.repeat(fv, epochs),
             label='reference model')
    plt.xlabel("Epochs")
    plt.ylabel("Fair value")
    plt.legend()
    plt.savefig(f"{plot_fold}/training_nn_maturity_{maturity:.3f}.png", bbox_inches='tight')
    plt.clf()

    print("neural network optimization ended")

    # # Drawing the vector field theta
    # x = np.arange(-1.7, 1.85, 0.15)
    # y = np.arange(-1.7, 1.85, 0.15)
    # xv, yv = np.meshgrid(x, y)
    # theta = i_theta_mart.theta(torch.stack([torch.from_numpy(xv), torch.from_numpy(yv)], dim=2)).detach().numpy()
    #
    # # plotting the loss function
    # xloss = np.arange(-1.7, 1.71, 0.01)
    # yloss = loss.cost(torch.from_numpy(xloss))
    #
    # # Depict illustration
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # CS = ax.contour(xlossv, ylossv, zloss, cmap='viridis', levels=7)
    # ax.clabel(CS, inline=True, fontsize=10)
    # ax.quiver(xv, yv, theta[:, :, 0], theta[:, :, 1], color='g')
    # # plt.streamplot(xv, yv, theta[:, :, 0], theta[:, :, 1], density=1.4, linewidth=None, color='#A23BEC')
    # ax.plot(epicenter[0], epicenter[1], 'rh')
    # # plt.title(f'Parametric optimizer for uncertainty level h={uncertainty_level}')
    # plt.savefig(f"{plot_fold}/nn_optimizer_level_{uncertainty_level:.5f}.png", bbox_inches='tight')
    #
    # print(f"elaboration time: {time.time() - start:.2f} seconds")

    # # let's see the parameters of the model
    # print("-----------------------------------------------")
    # for name, param in i_theta.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param} \n")
    # print("-----------------------------------------------")