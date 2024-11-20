import sys
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

    # This main performs the neural network optimization to compute the functional I_Theta with gaussian
    # reference measure and double dome loss function at a specific uncertainty level (uncertainty_level)
    # it is used to obtain the plots of Section 4.4

    torch.set_default_dtype(torch.float64)
    # ---------------------- Inputs ----------------------
    plot_fold = 'double_dome'
    eval_type = 'both'  # choose among ['neural_network','one_dimensional','both']
    city_centers = [[0., 0.], [1.25, 0.]]
    cost_level_1 = 0.5
    radius_1 = 1.
    cost_level_2 = 0.3
    radius_2 = 0.75
    epicenter = torch.tensor([0.75, 0.25])
    variance = 1.
    uncertainty_level = 0.5
    net_width = 20
    net_depth = 4
    mc_samples = 2**15
    mc_samples_1d = 2**16
    learning_rate = 0.001
    learning_rate_1d = 0.001
    epochs = 1100
    rolling_window = 100
    epochs_1d = epochs
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
                       f'\ncity center 2: {city_centers[1]}, cost level 2: {cost_level_2}, radius 2: {radius_2}'
                       f'\nepicenter earthquake: {epicenter}, variance: {variance}')
    dump_summary.write(f'\nuncertainty level: {uncertainty_level}')
    dump_summary.write(f'\nneural network depth: {net_depth}\nneural network width: {net_width}'
                       f'\nmc saples: {mc_samples}\nlearning rate: {learning_rate}\nepochs: {epochs}'
                       f'\nrolling window: {rolling_window}')
    if eval_type in ['one_dimensional', 'both']:
        dump_summary.write(f'\n----------- 1-d optimization --------------'
                           f'\nmc samples: {mc_samples_1d}\nlearning rate: {learning_rate_1d}\nepochs: {epochs_1d}'
                           f'\n-------------------------------------------')

    # fixing the seed
    torch.manual_seed(29)
    np.random.seed(3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    start = time.time()

    # initialization of loss function, baseline model, and penalisation function
    loss = cf.DoubleGaussianKernelCost2DTensor(x_1=city_centers[0][0], y_1=city_centers[0][1],
                                               x_2=city_centers[1][0], y_2=city_centers[1][1],
                                               level_1=cost_level_1, radius_1=radius_1,
                                               level_2=cost_level_2, radius_2=radius_2)
    mu = torch.distributions.multivariate_normal.MultivariateNormal(epicenter, variance * torch.eye(2))
    penalty = pnl.PolyPenaltyTensor(2)

    # computing the expected loss via montecarlo
    exp_samples = 1000000
    rnd_y = mu.sample([exp_samples])
    expected_loss = torch.mean(loss.cost(rnd_y[:, 0], rnd_y[:, 1]))
    exp_loss_mc_err = 2.33 * torch.std(loss.cost(rnd_y[:, 0], rnd_y[:, 1])) / np.sqrt(exp_samples)
    print("-------------------------------------")
    print(f"Expected Loss: {expected_loss:.6f} --> [{expected_loss - exp_loss_mc_err:.6f}, {expected_loss + exp_loss_mc_err:.6f}]")

    # computing the worst case loss at the uncertainty level h via one dimensional optimization
    if eval_type in ['both', 'one_dimensional']:

        print("starting one dimensional optimization...")
        # initializing the one dimensional neural network
        i_theta_1d = nnutils.ITheta_1d(func= loss.cost, grad=loss.gradient, penalty=penalty.evaluate, mu=mu,
                               h=uncertainty_level, nr_sample=mc_samples_1d)
        optm = torch.optim.Adam(i_theta_1d.parameters(), lr=learning_rate_1d)
        ws_loss_1d = []
        for i in range(epochs_1d):
            ws_loss = nnutils.train(i_theta_1d, optm)
            ws_loss_1d.append(-float(ws_loss))
        ws_loss_1d = pd.Series(ws_loss_1d).rolling(rolling_window).mean()

        print(f"Asymptotic worst case loss: {ws_loss_1d[ws_loss_1d.shape[0]-1]:.6f}")
        dump_summary.write(f'\nWorst case loss through asymptotic optimizer: {ws_loss_1d[ws_loss_1d.shape[0]-1]:.8f}')
        for name, param in i_theta_1d.named_parameters():
            print(f"Asymptotic optimizer theta: {param:.8f}")

        # plotting the training phase for the 1d optimization
        plt.plot(np.arange(1, epochs_1d + 1), ws_loss_1d, label='asymptotic optimizer')
        plt.savefig(os.path.join(plot_fold, f"training_1d_level_{uncertainty_level:.5f}.png"), bbox_inches='tight')
        plt.clf()

        print("One dimensional optimization ended")

    # computing the worst case loss at the uncertainty level h via nn optimization
    if eval_type in ['both', 'neural_network']:

        print("starting neural network optimization...")
        #initializing the neural network
        i_theta = nnutils.ITheta(func=loss.cost, penalty=penalty.evaluate, mu=mu,
                         h=uncertainty_level, width=net_width, depth=net_depth,
                         nr_sample=mc_samples)
        optm = torch.optim.Adam(i_theta.parameters(), lr=learning_rate)
        out_vector = []
        for i in range(epochs):
            out = nnutils.train(i_theta, optm)
            out_vector.append(-float(out))
        out_vector = pd.Series(out_vector).rolling(rolling_window).mean()

        print(f"Parametrized worst case loss: {out_vector[out_vector.shape[0]-1]:.6f}")
        dump_summary.write(f'\nWorst case loss through neural network optimizer: {out_vector[out_vector.shape[0]-1]:.8f}')

        # plotting the training phase
        plt.plot(np.arange(1, out_vector.shape[0] + 1), out_vector, label='neural network optimizer')
        if eval_type in ['both', 'one_dimensional']:
            plt.plot(np.arange(1, out_vector.shape[0] + 1), np.repeat(ws_loss_1d[ws_loss_1d.shape[0] - 1], epochs), label='asymptotic optimizer')
        plt.xlabel("Epochs")
        plt.ylabel("Worst case loss")
        plt.legend()
        plt.savefig(os.path.join(plot_fold, f"training_nn_level_{uncertainty_level:.5f}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(plot_fold, f"training_nn_level_{uncertainty_level:.5f}.eps"), format='eps', bbox_inches='tight')
        plt.clf()

        print("neural network optimization ended")

        # Drawing the vector field theta
        x = np.arange(-1.2, 2.2, 0.1)
        y = np.arange(-1.2, 1.3, 0.1)
        xv, yv = np.meshgrid(x, y)
        theta = i_theta.theta(torch.stack([torch.from_numpy(xv), torch.from_numpy(yv)], dim=2)).detach().numpy()

        # drawing the contour plot of the loss function
        xloss = np.arange(-1.2, 2.2, 0.01)
        yloss = np.arange(-1.2, 1.2, 0.01)
        xlossv, ylossv = np.meshgrid(xloss, yloss)
        zloss = loss.cost(torch.from_numpy(xlossv), torch.from_numpy(ylossv))

        # Depict illustration
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        CS = ax.contour(xlossv, ylossv, zloss, cmap='viridis', levels=7)
        ax.clabel(CS, inline=True, fontsize=10)
        ax.quiver(xv, yv, theta[:, :, 0], theta[:, :, 1], color='g')
        ax.plot(epicenter[0], epicenter[1], 'rh')
        plt.savefig(os.path.join(plot_fold, f"nn_optimizer_level_{uncertainty_level:.5f}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(plot_fold, f"nn_optimizer_level_{uncertainty_level:.5f}.eps"), format='eps', bbox_inches='tight')

        print(f"elaboration time: {time.time() - start:.2f} seconds")