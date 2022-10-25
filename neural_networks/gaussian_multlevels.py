import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import cost_functions as cf
import penalties as pnl
import utils as ut
import neural_networks.nn_utils as nnutils


if __name__ == '__main__':

    # This main performs the neural network optimization to compute the functional I_Theta with gaussian
    # reference measure for several uncertainty levels (h_levels)
    # it is used to obtain the plots of Section 4.3

    torch.set_default_dtype(torch.float64)
    # ---------------------- Inputs ----------------------
    plot_fold = 'gaussian_mult'
    eval_type = 'both'            # choose among ['neural_network','one_dimensional','both']
    city_center = [0., 0.]
    cost_level = 1.
    radius = 1.5
    epicenter = torch.tensor([1., 0.])
    variance = 1.
    h_levels = np.arange(0.001, 0.021, 0.001)
    principal_training_level = 0.03       # uncertainty level used to train the gradient of the function if neural_network_training=='simplified'
    neural_network_training = 'full'  # choose among ['full', 'simplified']
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

    # writing summary
    dump_summary = open(os.path.join(plot_fold, "summary.txt"), 'w')
    dump_summary.write(f'"eval type: {eval_type}'
                       f'\ncity center: {city_center}, cost level: {cost_level}, radius: {radius}'
                       f'\nepicenter earthquake: {epicenter}, variance: {variance}')
    if eval_type in ['neural_network', 'both']:
        dump_summary.write(
                       f'\nuncertainty levels: {h_levels.tolist()}'
                       f'\nneural network training: {neural_network_training}')
        if neural_network_training == 'simplified':
            dump_summary.write(f'\nprincipal training level: {principal_training_level}')
        dump_summary.write(f'\nneural network depth: {net_depth}\nneural network width: {net_width}'
                           f'\nmc saples: {mc_samples}\nlearning rate: {learning_rate}\nepochs: {epochs}'
                           f'\nrolling window: {rolling_window}')
    if eval_type in ['one_dimensional', 'both']:
        dump_summary.write(f'\n----------- 1-d optimization --------------'
                           f'\nmc samples: {mc_samples_1d}\nlearning rate: {learning_rate_1d}\nepochs: {epochs_1d}')


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

        dump_file = open(os.path.join(plot_fold, "1d_optim.txt"), "w")
        dump_file.write("Uncertainty levels: ")
        for h in h_levels:
            dump_file.write(f"{h:.4f},")

        loss_list = []
        param_list = []
        # cycling on the uncertainty levels
        print("starting one dimensional optimization...")
        for j in range(h_levels.shape[0]):

            # initializing the one dimesional neural network
            i_theta_1d = nnutils.ITheta_1d(func=loss.cost, grad=loss.gradient, penalty=penalty.evaluate, mu=mu,
                                   h=h_levels[j], nr_sample=mc_samples_1d)

            # computing the worst case loss at the uncertainty level h via one dimensional optimization
            optm = torch.optim.Adam(i_theta_1d.parameters(), lr=learning_rate_1d)
            ws_loss_1d = []
            for i in range(epochs_1d):
                ws_loss = nnutils.train(i_theta_1d, optm)
                ws_loss_1d.append(-float(ws_loss))
            ws_loss_1d = pd.Series(ws_loss_1d).rolling(rolling_window).mean()
            loss_list.append(ws_loss_1d[ws_loss_1d.shape[0]-1])
            for name, param in i_theta_1d.named_parameters():
                param_list.append(param)

            # plotting the training phase for the 1d optimization
            plt.plot(np.arange(1, epochs_1d + 1), ws_loss_1d, label='asymptotic optimizer')
            plt.savefig(os.path.join(plot_fold, f"training_1d_level_{h_levels[j]:.5f}.png"), bbox_inches='tight')
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
        dump_file_nn = open(os.path.join(plot_fold, "nn_optim.txt"), "w")
        dump_file_nn.write("Uncertainty levels: ")
        for h in h_levels:
            dump_file_nn.write(f"{h:.6f},")

        if neural_network_training == 'simplified':

            i_theta_reference = nnutils.ITheta(func=loss.cost, penalty=penalty.evaluate, mu=mu,
                             h=principal_training_level, width=net_width, depth=net_depth,
                             nr_sample=mc_samples)

            # computing the worst case loss at the chosen principal uncertainty level via nn optimization
            optm = torch.optim.Adam(i_theta_reference.parameters(), lr=learning_rate)
            for i in range(epochs):
                nnutils.train(i_theta_reference, optm)

        for j in range(h_levels.shape[0]):

            i_theta = nnutils.ITheta(func=loss.cost, penalty=penalty.evaluate, mu=mu,
                                   h=h_levels[j], width=net_width, depth=net_depth,
                                   nr_sample=mc_samples)

            if neural_network_training == 'simplified':
                i_theta.theta = copy.deepcopy(i_theta_reference.theta)
                for param in i_theta.theta.parameters():
                    param.requires_grad = False

            # computing the worst case loss at the uncertainty level h via nn optimization
            if neural_network_training == 'simplified':
                optm = torch.optim.Adam(filter(lambda p: p.requires_grad, i_theta.parameters()), lr=learning_rate_1d)
            else:
                optm = torch.optim.Adam(i_theta.parameters(), lr=learning_rate)

            out_vector = []
            for i in range(epochs):
                out = nnutils.train(i_theta, optm)
                out_vector.append(-float(out))
            out_vector = pd.Series(out_vector).rolling(rolling_window).mean()
            loss_nn_list.append(out_vector[out_vector.shape[0] - 1])

            # plotting the training phase
            plt.plot(np.arange(1, out_vector.shape[0] + 1), out_vector, label='neural network optimizer')
            if eval_type == 'both':
                plt.plot(np.arange(1, out_vector.shape[0] + 1), np.repeat(loss_list[j], epochs), label='asymptotic optimizer')
            # ----------------------------------------------------------
            plt.ylim(0.23, 0.2425) # reset if input data are changed!!!!
            # ----------------------------------------------------------
            plt.xlabel("Epochs")
            plt.ylabel("Worst case loss")
            plt.legend()
            plt.savefig(os.path.join(plot_fold, f"training_nn_level_{h_levels[j]:.5f}.png"), bbox_inches='tight')
            plt.clf()

            # Drawing the vector field theta
            x = np.arange(-1.69, 1.82, 0.13)
            y = np.arange(-1.69, 1.82, 0.13)
            xv, yv = np.meshgrid(x, y)
            theta = i_theta.theta(torch.stack([torch.from_numpy(xv), torch.from_numpy(yv)], dim=2)).detach().numpy()

            # drawing the contour plot of the loss function
            xloss = np.arange(-1.7, 1.71, 0.01)
            yloss = np.arange(-1.7, 1.71, 0.01)
            xlossv, ylossv = np.meshgrid(xloss, yloss)
            zloss = loss.cost(torch.from_numpy(xlossv), torch.from_numpy(ylossv))

            # Depict illustration
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            CS = ax.contour(xlossv, ylossv, zloss, cmap='viridis', levels=7)
            ax.clabel(CS, inline=True, fontsize=10)
            ax.quiver(xv, yv, theta[:, :, 0], theta[:, :, 1], color='g')
            ax.plot(epicenter[0], epicenter[1], 'rh')
            plt.savefig(os.path.join(plot_fold, f"nn_optimizer_level_{h_levels[j]:.5f}.png"), bbox_inches='tight')
            plt.clf()

        dump_file_nn.write("\nWorst case losses: ")
        for ls in loss_nn_list:
            dump_file_nn.write(f"{ls:.8f},")

        print("neural network optimization ended")

        if eval_type == 'both':
            x = [0.] + h_levels.tolist()
            loss_nn_list = [expected_loss] + loss_nn_list
            plt.plot(x, loss_nn_list, label='neural network optimization')
            loss_list = [expected_loss] + loss_list
            plt.plot(x, loss_list, label='asymptotic optimization')
            plt.xlabel("Uncertainty level")
            plt.ylabel("Worst case loss")
            plt.legend()
            plt.savefig(os.path.join(plot_fold, f"worst_case_loss_gaussian.png"), bbox_inches='tight')

    print(f"elaboration time: {(time.time() - start)/60.:.2f} minutes")