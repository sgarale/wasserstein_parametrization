import sys
import copy
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add the parent folder to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import cost_functions as cf
import penalties as pnl
import utils as ut
import neural_networks.nn_utils as nnutils

if __name__ == '__main__':

    # This main performs the transfer learning exercise of Section 4.4

    torch.set_default_dtype(torch.float64)
    # ---------------------- Inputs ----------------------
    plot_fold = 'double_dome_transfer'
    city_centers = [[0., 0.], [1.25, 0.]]
    cost_level_1 = 0.5
    radius_1 = 1.
    cost_level_2 = 0.3
    radius_2 = 0.75
    epicenter = torch.tensor([0.75, 0.25])
    variance = 1.
    epicenter_transfered = [torch.tensor([0.75, -0.25]), torch.tensor([1.5, 0.15]), torch.tensor([1.5, -0.15]), torch.tensor([-0.75, 0.5]), torch.tensor([-0.75, -0.5])]
    variance_transfered = [1., 1., 1., 1., 1.]
    uncertainty_level = 0.5
    net_width = 20
    net_depth = 4
    mc_samples_reference = 2**16
    mc_samples = 2**16
    mc_samples_transfer = 2**12
    learning_rate = 0.001
    learning_rate_transfer = 0.001
    epochs = 1100
    rolling_window = 100
    epochs_transfer = epochs
    # ----------------------------------------------------

    plot_fold = os.path.join("plots", plot_fold)
    ut.check_dir(plot_fold)

    # setting latex style for plots
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 13
    plt.rcParams['legend.fontsize'] = 13

    # writing summary
    dump_summary = open(os.path.join(plot_fold, "summary.txt"), 'w')
    dump_summary.write(f'city center 1: {city_centers[0]}, cost level 1: {cost_level_1}, radius 1: {radius_1}'
                       f'\ncity center 2: {city_centers[1]}, cost level 2: {cost_level_2}, radius 2: {radius_2}')
    dump_summary.write(f'\nuncertainty level: {uncertainty_level}')
    dump_summary.write(f'\nneural network depth: {net_depth}\nneural network width: {net_width}'
                       f'\nmc samples reference model: {mc_samples_reference}'
                       f'\nmc saples: {mc_samples}\nlearning rate: {learning_rate}\nepochs: {epochs}'
                       f'\nrolling window: {rolling_window}')
    dump_summary.write(f'\n----------- transfer optimization --------------'
                       f'\nmc samples: {mc_samples_transfer}\nlearning rate: {learning_rate_transfer}\nepochs: {epochs_transfer}')

    # fixing the seed
    torch.manual_seed(29)
    np.random.seed(3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    start = time.time()

    # initialization of loss function, baseline model, and penalization function
    loss = cf.DoubleGaussianKernelCost2DTensor(x_1=city_centers[0][0], y_1=city_centers[0][1],
                                               x_2=city_centers[1][0], y_2=city_centers[1][1],
                                               level_1=cost_level_1, radius_1=radius_1,
                                               level_2=cost_level_2, radius_2=radius_2)
    mu = torch.distributions.multivariate_normal.MultivariateNormal(epicenter, variance * torch.eye(2))
    penalty = pnl.PolyPenaltyTensor(2)

    print("Training of the reference neural network...")
    # computing the worst case loss at the uncertainty level h via nn optimization
    i_theta_reference = nnutils.ITheta(func=loss.cost, penalty=penalty.evaluate, mu=mu,
                                     h=uncertainty_level, width=net_width, depth=net_depth,
                                     nr_sample=mc_samples_reference)

    # computing the worst case loss at the chosen principal reference measure via nn optimization
    optm = torch.optim.Adam(i_theta_reference.parameters(), lr=learning_rate)
    out_reference = []
    for i in range(epochs):
        out = nnutils.train(i_theta_reference, optm)
        out_reference.append(-float(out))
    out_reference = pd.Series(out_reference).rolling(rolling_window).mean()

    dump_summary.write(f'\n\n------------------ Reference training ------------------------'
                       f'\nEpicenter: {epicenter}\t variance: {variance}'
                       f'\nWorst case loss: {out_reference[out_reference.shape[0] - 1]:.8f}')


    for j in range(len(epicenter_transfered)):

        print(f"Full neural network optimization at exercise {j + 1}/{len(epicenter_transfered)}...")

        # initializing the new reference measure
        mu = torch.distributions.multivariate_normal.MultivariateNormal(epicenter_transfered[j], variance_transfered[j] * torch.eye(2))
        # initializing the neural network
        i_theta = nnutils.ITheta(func=loss.cost, penalty=penalty.evaluate, mu=mu,
                         h=uncertainty_level, width=net_width, depth=net_depth,
                         nr_sample=mc_samples)
        # initializing the optimizer and training phase
        optm = torch.optim.Adam(i_theta.parameters(), lr=learning_rate)
        out_full = []
        for i in range(epochs):
            out = nnutils.train(i_theta, optm)
            out_full.append(-float(out))
        out_full = pd.Series(out_full).rolling(rolling_window).mean()

        dump_summary.write(f'\n-------------------- Exercise {j + 1} ---------------------'
                           f'\nEpicenter: {epicenter_transfered[j]}\tVariance: {variance_transfered[j]}'
                           f'\nWorst case loss full training:\t{out_full[out_full.shape[0]-1]:.8f}')

        print(f"Pretrained neural network optimization...")
        # initializing the pretrained network
        i_theta_transfer = nnutils.ITheta(func=loss.cost, penalty=penalty.evaluate, mu=mu,
                         h=uncertainty_level, width=net_width, depth=net_depth,
                         nr_sample=mc_samples)
        i_theta_transfer.theta = copy.deepcopy(i_theta_reference.theta)
        for param in i_theta_transfer.theta.parameters():
            param.requires_grad = False

        # initializing the optimizer and training phase for the pretrained network
        optm = torch.optim.Adam(filter(lambda p: p.requires_grad, i_theta_transfer.parameters()), lr=learning_rate_transfer)
        out_transfer = []
        for i in range(epochs_transfer):
            out = nnutils.train(i_theta_transfer, optm)
            out_transfer.append(-float(out))
        out_transfer = pd.Series(out_transfer).rolling(rolling_window).mean()

        dump_summary.write(f'\nWorst case loss partial training:\t{out_transfer[out_transfer.shape[0]-1]:.8f}')

        # plotting the training phase for the full and the partially trained networks
        plt.plot(np.arange(1, out_full.shape[0] + 1), out_full, label='full trained network')
        plt.plot(np.arange(1, out_transfer.shape[0] + 1), out_transfer, label='pretrained network')
        plt.xlabel("Epochs")
        plt.ylabel("Worst case loss")
        plt.legend()
        plt.savefig(os.path.join(plot_fold, f"transfer_exercise_{j + 1}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(plot_fold, f"transfer_exercise_{j + 1}.eps"), format='eps', bbox_inches='tight')
        plt.clf()

    print(f"elaboration time: {(time.time() - start)/60.:.2f} minutes")