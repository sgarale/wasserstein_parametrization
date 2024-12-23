import sys
import os
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch

# Add the parent folder to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import cost_functions as cf
import penalties as pnl
import utils as ut
import neural_networks.nn_utils as nnutils


def blackscholes_option(S, K, T, r, q, sigma, flag='Call'):
    """
    Computes the analytical price of call (or put) option using the GBM model (Black-Scholes formula).
    :param S: float, initial level of the asset
    :param K: float, strike level
    :param T: float, maturity
    :param r: float, zero coupon risk-free rate
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

def bull_call_spread(S, K_long, K_short, T, r, q, sigma):
    """
    Computes the analytical price of a bull call spread using the GBM model (Black-Scholes formula).
    :param S: float, initial level of the asset
    :param K_long: float, strike of the long call option
    :param K_short: float, strike of the short call option
    :param T: float, maturity
    :param r: float, zero coupon risk-free rate
    :param q: float, dividend rate
    :param sigma: float, volatility
    :param flag: str, available choices: "Call", "Put"
    :return: float
    """
    return blackscholes_option(S, K_long, T, r, q, sigma) - blackscholes_option(S, K_short, T, r, q, sigma)



if __name__ == '__main__':

    # This main performs the neural network optimization to compute robust bounds for the price of a bull call spread
    # in the martingale constraint setting. It is used to obtain the plot of Section 4.6.

    torch.set_default_dtype(torch.float64)
    # ---------------------- Inputs ----------------------
    plot_fold = 'bull_spread'
    p = 3
    drift = 0.
    volatility = 0.20
    penalty_power_growth = 5
    strike_long = 1.
    strike_short = 1.2
    T0 = 1.
    maturities = torch.arange(3, 33, 3) / 365.2425
    net_width = 20
    net_depth = 4
    mc_samples = 2**16
    mc_out_of_samples = 2**19
    learning_rate = 0.001
    epochs = 1100
    rolling_window = 100
    # ----------------------------------------------------

    plot_fold = os.path.join('plots', plot_fold)
    ut.check_dir(plot_fold)

    # setting latex style for plots
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 13
    plt.rcParams['legend.fontsize'] = 13

    # writing summary
    dump_summary = open(os.path.join(plot_fold, "summary.txt"), 'w')
    dump_summary.write(f'Order: {p}, strike lower: {strike_long}, strike upper: {strike_short},'
                       f'\ndrift: {drift}, volatility: {volatility}'
                       f'\nmaturities: {(torch.tensor(T0) + maturities).tolist()}'
                       f'\npower growth of the penalty function: {penalty_power_growth}')
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
    loss = cf.BullSpread(strike_long, strike_short)
    mu = torch.distributions.log_normal.LogNormal((drift - volatility * volatility / 2) * T0, volatility * math.sqrt(T0))
    penalty = pnl.RescaledPolyPenalty(sigma=volatility, n=penalty_power_growth)

    # computing the option fair value via Black-Scholes formula
    fv = bull_call_spread(S=1., K_long=strike_long, K_short=strike_short, T=T0, r=0., q=0., sigma=volatility)
    print(f'Fair value at time T={T0:.4f}: {fv:.4f}')
    dump_summary.write(f'\n\nFair value at time T=1: {fv:.4f}')

    expected_vector = [fv]
    upper_vector = [fv]
    lower_vector = [fv]

    for h in maturities:

        print(f'Maturity: {T0 + h:.4f}')
        dump_summary.write(f'\n\nMaturity: {T0+h:.4f}')

        # computing the option fair value via Black&Scholes formula
        fv = bull_call_spread(S=1., K_long=strike_long, K_short=strike_short, T=T0 + h, r=0., q=0., sigma=volatility)
        print(f'Fair value at time T={T0 + h:.4f}: {fv:.4f}')
        dump_summary.write(f'\n\nFair value at time T={T0 + h:.4f}: {fv:.4f}')
        expected_vector.append(float(fv))

        # computing the upper fair value bound at the uncertainty level h via nn optimization
        print("starting neural network optimization for the upper bound...")
        #initializing the neural network
        i_theta_mart = nnutils.IThetaMart(func=loss.cost, penalty=penalty, p=p, mu=mu,
                         h=h, width=net_width, depth=net_depth,
                         nr_sample=mc_samples)

        optm = torch.optim.Adam(i_theta_mart.parameters(), lr=learning_rate)
        out_vector = []
        for i in range(epochs):
            optm.zero_grad()
            y = mu.sample([i_theta_mart.nr_sample,1])
            out = i_theta_mart(y)
            out.backward()
            optm.step()
            out_vector.append(-float(out))
        train_hist = pd.Series(out_vector).rolling(rolling_window).mean()
        i_theta_mart.eval()
        y = mu.sample([mc_out_of_samples,1])
        upper_bound = -i_theta_mart(y)
        upper_vector.append(upper_bound.item())
        print(f"Parametrized upper bound: {upper_bound:.6f}")
        dump_summary.write(f'\nUpper bound through neural network optimizer: {upper_bound:.8f}')

        # plotting the training phase
        plt.plot(np.arange(1, train_hist.shape[0] + 1), train_hist, label='neural network upper model')
        plt.plot(np.arange(1, train_hist.shape[0] + 1), np.repeat(fv, epochs),
                 label='reference model')
        plt.xlabel("Epochs")
        plt.ylabel("Fair value")
        plt.legend()
        plt.savefig(os.path.join(plot_fold, f'training_nn_upper_maturity_{T0 + h:.3f}.png'), bbox_inches='tight')
        plt.clf()

        print(f"neural network optimization for the upper bound ended")

        # computing the lower fair value bound at the uncertainty level h via nn optimization
        print("starting neural network optimization for the lower bound...")
        # initializing the neural network
        i_theta_mart = nnutils.IThetaMart(func=loss.cost, penalty=penalty, p=p, mu=mu,
                                  h=h, width=net_width, depth=net_depth,
                                  nr_sample=mc_samples, sup=False)

        optm = torch.optim.Adam(i_theta_mart.parameters(), lr=learning_rate)
        out_vector = []
        for i in range(epochs):
            optm.zero_grad()
            y = mu.sample([i_theta_mart.nr_sample,1])
            out = i_theta_mart(y)
            out.backward()
            optm.step()
            out_vector.append(float(out))
        train_hist = pd.Series(out_vector).rolling(rolling_window).mean()
        i_theta_mart.eval()
        y = mu.sample([mc_out_of_samples,1])
        lower_bound = i_theta_mart(y)
        lower_vector.append(lower_bound.item())
        print(f"Parametrized lower bound: {lower_bound:.6f}")
        dump_summary.write(
            f'\nLower bound through neural network optimizer: {lower_bound:.8f}')

        # plotting the training phase
        plt.plot(np.arange(1, train_hist.shape[0] + 1), train_hist, label='neural network lower model')
        plt.plot(np.arange(1, train_hist.shape[0] + 1), np.repeat(fv, epochs),
                 label='reference model')
        plt.xlabel("Epochs")
        plt.ylabel("Fair value")
        plt.legend()
        plt.savefig(os.path.join(plot_fold,f"training_nn_lower_maturity_{T0 + h:.3f}.png"), bbox_inches='tight')
        plt.clf()

        print(f"neural network optimization for lower bound ended")

    maturities = torch.tensor(T0) + maturities
    maturities = torch.concat([torch.tensor([T0]), maturities])
    plt.plot(maturities, upper_vector, label='Upper level')
    plt.plot(maturities, lower_vector, label='Lower level')
    plt.plot(maturities, expected_vector, label='B\&S fair value')
    plt.xlabel('Maturity')
    plt.ylabel('Fair value')
    # --------------- reset if input parameters are changed ------------------
    plt.xlim([maturities[0] - 1. / 365.2425, maturities[-1] + 1. / 365.2425])
    plt.ylim([0.048, 0.0670])
    # ------------------------------------------------------------------------
    plt.legend()
    plt.savefig(os.path.join(plot_fold,'option_levels.png'), bbox_inches='tight')
    plt.savefig(os.path.join(plot_fold,'option_levels.eps'), format='eps', bbox_inches='tight')
    plt.clf()

    dump_summary.write(f'\n\nB&S prices: {expected_vector}')
    dump_summary.write(f'\nUpper prices: {upper_vector}')
    dump_summary.write(f'\nLower prices: {lower_vector}')

    print(f'\nTotal time for the evaluation: {(time.time() - start)/60.:.2f} m')