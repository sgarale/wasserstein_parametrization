import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import cost_functions as cf
import penalties as pnl
import gaussian_full as gauss
import time


if __name__ == '__main__':

    torch.set_default_dtype(torch.float64)
    # ---------------------- Inputs ----------------------
    plot_fold = 'gaussian_mult_simpl'
    eval_type = 'neural_network'            # choose among ['neural_network','one_dimensional','both']
    city_center = [0., 0.]
    cost_level = 1.
    radius = 1.5
    epicenter = torch.tensor([1., 0.])
    variance = 1.
    h_levels = np.arange(0.015, 0.315, 0.015)
    principal_training_level = 0.015            # uncertainty level used to train the gradient of the function
    neural_network_training = 'simplified'  # choose among ['full', 'simplified']
    net_width = 20
    net_depth = 2
    mc_samples = 2**14
    mc_samples_1d = 2**15
    learning_rate = 0.002
    learning_rate_1d = 0.005
    epochs = 1100
    rolling_window = 100
    epochs_1d = epochs
    # ----------------------------------------------------

    plot_fold = f"plots/{plot_fold}"
    gauss.check_dir(plot_fold)

    # fixing the seed
    torch.manual_seed(29)
    np.random.seed(3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    start = time.time()

    # initialization of loss function, baseline model, and penalisation function
    loss = cf.GaussianKernelCost2DTensor(x_0=city_center[0], y_0=city_center[1], level=cost_level, radius=radius)
    mu = torch.distributions.multivariate_normal.MultivariateNormal(epicenter, variance * torch.eye(2))
    penalty = pnl.PolyPenaltyTensor(2)

    # computing the expected loss via montecarlo
    exp_samples = 1000000
    rnd_y = mu.sample([exp_samples])
    expected_loss = torch.mean(loss.cost(rnd_y[:, 0], rnd_y[:, 1]))
    exp_loss_mc_err = 2.33 * torch.std(loss.cost(rnd_y[:, 0], rnd_y[:, 1])) / np.sqrt(exp_samples)
    print("-------------------------------------")
    print(
        f"Expected Loss: {expected_loss:.4f} --> [{expected_loss - exp_loss_mc_err:.4f}, {expected_loss + exp_loss_mc_err:.4f}]")


    if eval_type in ['one_dimensional', 'both']:

        dump_file = open(plot_fold + "/1d_optim.txt", "w")
        dump_file.write("Uncertainty levels: ")
        for h in h_levels:
            dump_file.write(f"{h:.4f},")

        loss_list = []
        param_list = []
        # cycling on the uncertainty levels
        print("starting one dimensional optimization...")
        for j in range(h_levels.shape[0]):
            # initializing the objects: penalisation function, loss function, baseline measure, operators
            i_theta_1d = gauss.ITheta_1d(func=loss.cost, grad=loss.gradient, penalty=penalty.evaluate, mu=mu,
                                   h=h_levels[j], nr_sample=mc_samples_1d)

            # computing the worst case loss at the uncertainty level h via one dimensional optimization
            optm = torch.optim.Adam(i_theta_1d.parameters(), lr=learning_rate_1d)
            ws_loss_1d = []
            for i in range(epochs_1d):
                ws_loss = gauss.train(i_theta_1d, optm)
                ws_loss_1d.append(-float(ws_loss))
            ws_loss_1d = pd.Series(ws_loss_1d).rolling(rolling_window).mean()
            loss_list.append(ws_loss_1d[ws_loss_1d.shape[0]-1])
            for name, param in i_theta_1d.named_parameters():
                param_list.append(param)

            # plotting the training phase for the 1d optimization
            plt.plot(np.arange(1, epochs_1d + 1), ws_loss_1d, label='1d optim')
            plt.savefig(f"{plot_fold}/training_1d_level_{h_levels[j]:.3f}.png", bbox_inches='tight')
            plt.clf()
        dump_file.write("\nWorst case losses: ")
        for ls in loss_list:
            dump_file.write(f"{ls:.8f},")
        dump_file.write("\nOptimizers: ")
        for param in param_list:
            dump_file.write(f"{param:.8f},")
        print("One dimensional optimization ended")

    loss_nn_list = []
    if eval_type in ['neural_network', 'both']:

        print(f"starting neural network {neural_network_training} optimization...")
        dump_file_nn = open(plot_fold + "/nn_optim.txt", "w")
        dump_file_nn.write("Uncertainty levels: ")
        for h in h_levels:
            dump_file_nn.write(f"{h:.4f},")

        if neural_network_training == 'simplified':

            i_theta_reference = gauss.ITheta(func=loss.cost, penalty=penalty.evaluate, mu=mu,
                             h=principal_training_level, width=net_width, depth=net_depth,
                             nr_sample=mc_samples)

            # computing the worst case loss at the chosen principal uncertainty level via nn optimization
            optm = torch.optim.Adam(i_theta_reference.parameters(), lr=learning_rate)

        for j in range(h_levels.shape[0]):

            i_theta = gauss.ITheta(func=loss.cost, penalty=penalty.evaluate, mu=mu,
                                   h=h_levels[j], width=net_width, depth=net_depth,
                                   nr_sample=mc_samples)

            if neural_network_training == 'simplified':
                i_theta.theta = i_theta_reference.theta
                for param in i_theta.theta.parameters():
                    param.requires_grad = False

            # computing the worst case loss at the uncertainty level h via nn optimization
            if neural_network_training == 'simplified':
                optm = torch.optim.Adam(filter(lambda p: p.requires_grad, i_theta.parameters()), lr=learning_rate)
            else:
                optm = torch.optim.Adam(i_theta.parameters(), lr=learning_rate)

            out_vector = []
            for i in range(epochs):
                out = gauss.train(i_theta, optm)
                out_vector.append(-float(out))
            out_vector = pd.Series(out_vector).rolling(rolling_window).mean()
            loss_nn_list.append(out_vector[out_vector.shape[0] - 1])

            # plotting the training phase
            plt.plot(np.arange(1, out_vector.shape[0] + 1), out_vector, label='NN optimizer')
            if eval_type == 'both':
                plt.plot(np.arange(1, out_vector.shape[0] + 1), np.repeat(loss_list[j], epochs), label='Exact optimizer')
            plt.xlabel("Epochs")
            plt.ylabel("Worst case loss")
            plt.legend()
            plt.savefig(f"{plot_fold}/training_nn_level_{h_levels[j]:.3f}.png", bbox_inches='tight')
            plt.clf()

        dump_file_nn.write("\nWorst case losses: ")
        for ls in loss_nn_list:
            dump_file_nn.write(f"{ls:.8f},")

        print("neural network optimization ended")

        if eval_type == 'both':
            x = [0.] + h_levels.tolist()
            loss_list = [expected_loss] + loss_list
            loss_nn_list = [expected_loss] + loss_nn_list
            plt.plot(x, loss_list, label='1-d optimization')
            plt.plot(x, loss_nn_list, label='neural network optimization')
            plt.xlabel("Uncertainty level")
            plt.ylabel("Worst case loss")
            plt.legend()
            plt.savefig(f"{plot_fold}/worst_case_loss_gaussian.png", bbox_inches='tight')

    print(f"elaboration time: {(time.time() - start)/60.:.2f} minuti")