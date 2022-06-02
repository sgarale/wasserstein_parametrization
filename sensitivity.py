import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt


# Analytical prices call and put option
def blackscholes_option(S, K, T, r, q, sigma, flag='Call'):
    """
    It gives the analytical price of call and put option using the GBM model (Black-Scholes formula).
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

def bs_vega(S, K, T, r, q, sigma, flag='Call'):
    """
    it gives the analytical value of the vega, which is the sensitivity of a Black Scholes option
    with respect to the volatility.
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
        raise NotImplementedError("Greek Vega not yet implemented for Put options")
    else:
        raise Exception('Unknown option type.')

    d1 = (np.log(S / K) + (r - q + 0.5 * np.power(sigma, 2)) * T) / (sigma * np.sqrt(T))

    return np.exp(-q * T) * S * np.sqrt(T) * norm.pdf(d1)



if __name__=='__main__':

    # graphic of vega option when strikes varies
    S = np.arange(50, 180, 0.1)
    # S = 100.
    K = 100.
    # K = np.arange(60, 140, 0.1)
    T = .5
    r = 0.
    q = 0.
    sigma = 0.20
    plt.plot(S, bs_vega(S, K, T, r, q, sigma))
    plt.show()