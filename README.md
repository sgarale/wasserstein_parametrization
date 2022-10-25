## A parametric approach to the estimation of convex risk functionals based on Wasserstein distance
____________________________

This repository contains the implementation of the numerical examples in ["A parametric approach to the estimation of convex risk functionals based on Wasserstein distance"](link ad arxiv)

### Requirements
The file [requirements.txt](requirements.txt) contains minimal requirements for the environment necessary to run the scripts of this repository.

### Scripts

All the scripts with a "main" contain a brief description of their contents. At the beginning of each main it is possible to change the input parameters.
The code is written using an object oriented philosophy, so that generalizations should be feasible.

Here, a list of the scripts used to produce the plots of the paper:
- [cost_functions.py](cost_functions.py) produces the plot of the loss function for the earthquake model of Section 4.1. It also contains all the other loss functions used in the examples;
- [dirac_mult.py](finite_dim/dirac_mult.py) reproduces the finite-dimensional optimization described in Section 3.1. With the original inputs produces the plots of Section 4.2;
- [gaussian_multlevels.py](neural_networks/gaussian_multlevels.py) reproduces the neural network scheme described in Section 3.2. With the original inputs it produces the plots of Section 4.3;
- [double_dome_full.py](neural_networks/double_dome_full.py) produces the plot of the optimization in Section 4.4;
- [double_dome_transfer.py](neural_networks/double_dome_transfer.py) performs the transfer learning exercise described in Section 4.4. With the original inputs produces the plots of Section 4.4;
- [bull_spread.py](neural_networks/bull_spread.py) performs the neural network optimization under the martingale constraint as described in Section 3.3. With the original inputs produces the plot of Section 4.6.

Execution time may vary depending on the machine. We recommend to reduce the number of samples when first running the code.
In some cases, in order to reproduce the plots of the paper, particular settings for the limits of the plots are given. If the input parameters are changed, these settings should be reset (it is enough to comment out the corresponding lines).

####Version 0.9

### License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details