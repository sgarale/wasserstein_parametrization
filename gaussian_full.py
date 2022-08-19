import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import multivariate_normal
from scipy import optimize
import cost_functions as cf
import penalties as pnl



class ThetaReLUNetwork(nn.Module):

    def __init__(self, width, depth=1):
        super(ThetaReLUNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, width)
        )
        for i in range(depth - 1):
            self.linear_relu_stack.append(nn.ReLU())
            self.linear_relu_stack.append(nn.Linear(width, width))
        self.linear_relu_stack.append(nn.ReLU())
        self.linear_relu_stack.append(nn.Linear(width, 2))
        # self.linear_relu_stack.apply(init_weights)

    def forward(self, x):
        theta = self.linear_relu_stack(x)
        return theta

class ThetaSigmoidNetwork(nn.Module):

    def __init__(self, width, depth=1):
        super(ThetaSigmoidNetwork, self).__init__()
        self.linear_sigm_stack = nn.Sequential(
            nn.Linear(2, width)
        )
        for i in range(depth - 1):
            self.linear_sigm_stack.append(nn.Sigmoid())
            self.linear_sigm_stack.append(nn.Linear(width, width))
        self.linear_sigm_stack.append(nn.Sigmoid())
        self.linear_sigm_stack.append(nn.Linear(width, 2))

    def forward(self, x):
        theta = self.linear_sigm_stack(x)
        return theta


class PenalisedLoss(nn.Module):

    def __init__(self, func, penalty, h):
        super(PenalisedLoss, self).__init__()
        self.func = func
        self.penalty = penalty
        # self.p = p
        self.h = h

    def forward(self, y, theta_y):
        integral = torch.mean(self.func(y[:, 0] + theta_y[:, 0], y[:, 1] + theta_y[:, 1]))
        L2_norm_theta = torch.mean(torch.pow(theta_y, 2).sum(dim=1)).sqrt()
        penal_term = self.h * self.penalty(L2_norm_theta / self.h)
        return - (integral - penal_term)



class ITheta(nn.Module):

    def __init__(self, func, penalty, mu, h, width, depth, nr_sample):
        super(ITheta, self).__init__()
        self.theta = ThetaReLUNetwork(width, depth)
        self.mult_coeff = nn.Parameter(torch.tensor(.1))
        self.penalised_loss = PenalisedLoss(func=func, penalty=penalty, h=h)
        self.mu = mu
        self.nr_sample = nr_sample

    def forward(self):
        y = self.mu.sample([self.nr_sample])
        theta_y = self.mult_coeff * self.theta(y)
        i_theta = self.penalised_loss(y, theta_y)
        return i_theta



class PenalisedLoss_1d(nn.Module):

    def __init__(self, func, grad, penalty, h):
        super(PenalisedLoss_1d, self).__init__()
        self.func = func
        self.penalty = penalty
        self.grad = grad
        self.h = h

    def forward(self, y, theta):
        theta_y = theta * torch.stack(self.grad(y[:, 0], y[:, 1]), dim=1)
        integral = torch.mean(self.func(y[:, 0] + theta_y[:, 0], y[:, 1] + theta_y[:, 1]))
        L2_norm_theta = torch.mean(torch.pow(theta_y, 2).sum(dim=1)).sqrt()
        penal_term = self.h * self.penalty(L2_norm_theta / self.h)
        return - (integral - penal_term)



class ITheta_1d(nn.Module):

    def __init__(self, func, grad, penalty, mu, h, nr_sample):
        super(ITheta_1d, self).__init__()
        self.mult_coeff = nn.Parameter(torch.tensor(.5))
        self.penalised_loss = PenalisedLoss_1d(func=func, grad=grad, penalty=penalty, h=h)
        self.mu = mu
        self.nr_sample = nr_sample

    def forward(self):
        y = self.mu.sample([self.nr_sample])
        theta = self.mult_coeff
        i_theta = self.penalised_loss(y, theta)
        return i_theta


class ITheta_1dnp():

    def __init__(self, func, grad, penalty, mu, h, nr_sample):
        self.theta = .1
        self.func = func
        self.grad = grad
        self.penalty = penalty
        self.mu = mu
        self.h = h,
        self.nr_sample = nr_sample
        self.value = None

    def loss_to_optim(self, theta):
        y = self.mu.rvs(size=self.nr_sample)
        theta_y = theta * np.stack(self.grad(y[:, 0], y[:, 1]), axis=1)
        integral = np.mean(self.func(y[:, 0] + theta_y[:, 0], y[:, 1] + theta_y[:, 1]))
        L2_norm_theta = np.sqrt(np.mean(np.power(theta_y, 2).sum(axis=1)))
        penal_term = self.h * self.penalty(L2_norm_theta / self.h)
        res_tmp = - (integral - penal_term)
        return res_tmp[0]

    def optimize(self, x0=None):
        if x0 == None:
            x0 = self.theta
        res = optimize.basinhopping(self.loss_to_optim, x0=x0, stepsize=0.1, niter=50, niter_success=10)
        self.theta = res.x[0]
        # test_theta = np.arange(0.005, 1., 0.005)
        # test_values = [self.loss_to_optim(th) for th in test_theta]
        # self.theta = test_theta[np.argmin(test_values)]
        self.value = - self.loss_to_optim(self.theta)



def train(model, optimizer):
    model.zero_grad()
    output = model()
    output.backward()
    optimizer.step()
    return output



def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':

    # ---- Plots subfolder --------
    plot_fold = 'gaussian'
    # -----------------------------
    plot_fold = f"plots/{plot_fold}"
    check_dir(plot_fold)

    # fixing the seed
    torch.manual_seed(29)
    np.random.seed(3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    #################### ITheta tests #####################
    torch.set_default_dtype(torch.float64)
    city_center = [0., 0.]
    cost_level = 1.
    radius = 1.5
    epicenter = torch.tensor([1., 0.])
    uncertainty_level = 0.3
    net_width = 15
    net_depth = 2
    mc_samples = 2**15
    mc_samples_1d = 50000
    learning_rate = 0.005
    learning_rate_1d = 0.01
    epochs = 1050
    rolling_window = 50
    epochs_1d = epochs

    # initializing the objects: penalisation function, loss function, baseline measure, operators
    penalty = pnl.PolyPenaltyTensor(2)
    penalty_np = pnl.PolyPenalty(2)
    loss = cf.GaussianKernelCost2DTensor(x_0=city_center[0], y_0=city_center[1],
                                         level=cost_level, radius=radius)
    loss_np = cf.GaussianKernelCost2D(x_0=city_center[0], y_0=city_center[1],
                                         level=cost_level, radius=radius)
    mu = torch.distributions.multivariate_normal.MultivariateNormal(epicenter, torch.eye(2))
    mu_np = multivariate_normal(epicenter, np.eye(2))
    i_theta = ITheta(func=loss.cost, penalty=penalty.evaluate, mu=mu,
                     h=uncertainty_level, width=net_width, depth=net_depth,
                     nr_sample=mc_samples)
    i_theta_1d = ITheta_1d(func= loss.cost, grad=loss.gradient, penalty=penalty.evaluate, mu=mu,
                           h=uncertainty_level, nr_sample=mc_samples_1d)
    i_theta_1dnp = ITheta_1dnp(func=loss_np.cost, grad=loss_np.gradient, penalty=penalty_np.evaluate, mu=mu_np,
                           h=uncertainty_level, nr_sample=mc_samples_1d)

    # computing the expected loss via montecarlo
    rnd_y = mu.sample([1000000])
    expected_loss = torch.mean(loss.cost(rnd_y[:, 0], rnd_y[:, 1]))
    print("-------------------------------------")
    print(f"Expected Loss: {expected_loss:.4f}")

    # computing the worst case loss at the uncertainty level h via one dimensional optimization (via basinhopping)
    # i_theta_1dnp.optimize()
    # print(f"Exact worst case loss: {i_theta_1dnp.value:.4f}")
    # print(f"Exact worst case optimizer: {i_theta_1dnp.theta:.4f}")

    # # computing the worst case loss at the uncertainty level h via one dimensional optimization
    # optm = torch.optim.Adam(i_theta_1d.parameters(), lr=learning_rate_1d)
    # ws_loss_1d = []
    # for i in range(epochs_1d):
    #     ws_loss = train(i_theta_1d, optm)
    #     ws_loss_1d.append(-float(ws_loss))
    # print(f"Exact worst case loss: {-ws_loss:.4f}")
    # # plotting the training phase for the 1d optimization
    # plt.plot(np.arange(1, epochs + 1), ws_loss_1d, label='1d optim')

    # computing the worst case loss at the uncertainty level h via nn optimization
    optm = torch.optim.Adam(i_theta.parameters(), lr=learning_rate)
    out_vector = []
    for i in range(epochs):
        out = train(i_theta, optm)
        out_vector.append(-float(out))
    out_vector = pd.Series(out_vector).rolling(rolling_window).mean()
    print(f"Parametrized worst case loss: {out_vector[out_vector.shape[0]-1]:.4f}")
    # plotting the training phase
    plt.plot(np.arange(1, out_vector.shape[0] + 1), out_vector, label='nn optim')
    plt.plot(np.arange(1, out_vector.shape[0] + 1), np.repeat(0.2682, epochs), label='one-d optim')
    # ax = plt.gca()
    # x_left, x_right = ax.get_xlim()
    # y_low, y_high = ax.get_ylim()
    # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high)))
    # plt.title(f"Training of the worst case loss for h={uncertainty_level}")
    plt.xlabel("Epochs")
    plt.ylabel("Worst case loss")
    plt.legend()
    plt.savefig(f"{plot_fold}/gauss_training_level_{uncertainty_level}.png", bbox_inches='tight')
    plt.show()



    # Drawing the vector field theta
    x = np.arange(-1.7, 1.85, 0.15)
    y = np.arange(-1.7, 1.85, 0.15)
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
    # plt.streamplot(xv, yv, theta[:, :, 0], theta[:, :, 1], density=1.4, linewidth=None, color='#A23BEC')
    ax.plot(epicenter[0], epicenter[1], 'rh')
    # plt.title(f'Parametric optimizer for uncertainty level h={uncertainty_level}')
    plt.savefig(f"{plot_fold}/gauss_optimizer_level_{uncertainty_level}.png", bbox_inches='tight')
    plt.show()

    # # let's see the parameters of the model
    # print("-----------------------------------------------")
    # for name, param in i_theta.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param} \n")
    # print("-----------------------------------------------")