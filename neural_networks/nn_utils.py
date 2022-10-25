from torch import nn
import torch

# modules for two-dimensional neural networks and related parametric functional

class ThetaReLUNetwork(nn.Module):
    """
    ReLU network from R^2 to R^2
    """

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

    def forward(self, x):
        theta = self.linear_relu_stack(x)
        return theta


class ThetaSigmoidNetwork(nn.Module):
    """
    Sigmoid network from R^2 to R^2
    """

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
    """
    Loss for the optimization with Wasserstein-2
    """

    def __init__(self, func, penalty, h):
        super(PenalisedLoss, self).__init__()
        self.func = func
        self.penalty = penalty
        self.h = h

    def forward(self, y, theta_y):
        integral = torch.mean(self.func(y[:, 0] + theta_y[:, 0], y[:, 1] + theta_y[:, 1]))
        L2_norm_theta = torch.mean(torch.pow(theta_y, 2).sum(dim=1)).sqrt()
        penal_term = self.h * self.penalty(L2_norm_theta / self.h)
        return - (integral - penal_term)


class ITheta(nn.Module):
    """
    Parametric functional I_Theta for the unconstrained case
    """

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
    """
    Loss for the optimization with Wasserstein-2 along the gradient of the function f
    """

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
    """
    Parametric functional I_Theta for the unconstrained case with optimization along the gradient of the function f
    """

    def __init__(self, func, grad, penalty, mu, h, nr_sample):
        super(ITheta_1d, self).__init__()
        self.mult_coeff = nn.Parameter(torch.tensor(.3))
        self.penalised_loss = PenalisedLoss_1d(func=func, grad=grad, penalty=penalty, h=h)
        self.mu = mu
        self.nr_sample = nr_sample

    def forward(self):
        y = self.mu.sample([self.nr_sample])
        theta = self.mult_coeff
        i_theta = self.penalised_loss(y, theta)
        return i_theta



# modules for one-dimensional neural networks and related parametric functional

class MartReLUNetwork(nn.Module):
    """
    ReLU network from R to R
    """

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
    """
    Loss for the optimization with martingale constraint in R
    """

    def __init__(self, func, penalty, p, h, sup=True):
        super(PenalisedLossMart, self).__init__()
        self.func = func
        self.penalty = penalty
        self.p = p
        self.h = h
        self.sign = torch.tensor(1)
        if sup:
            self.sign = torch.tensor(-1)

    def forward(self, y, theta_y):
        integral = torch.mean((self.func(y + theta_y) + self.func(y - theta_y)) / 2)
        Lp_norm_theta = torch.pow(torch.mean(torch.pow(torch.abs(theta_y), self.p)), 1./self.p)
        penal_term = self.h * self.penalty.evaluate(torch.pow(Lp_norm_theta, 2) / self.h)
        return self.sign * (integral + self.sign * penal_term)


class IThetaMart(nn.Module):
    """
    Parametric functional I_Theta for the martingale constraint case in R
    """

    def __init__(self, func, penalty, p, mu, h, width, depth, nr_sample, sup=True):
        super(IThetaMart, self).__init__()
        self.theta = MartReLUNetwork(width, depth)
        self.mult_coeff = None
        self.penalised_loss = PenalisedLossMart(func=func, penalty=penalty, p=p, h=h, sup=sup)
        self.mu = mu
        self.nr_sample = nr_sample
        self.scale_norm()

    def forward(self):
        y = self.mu.sample([self.nr_sample, 1])
        theta_y = self.mult_coeff * self.theta(y)
        i_theta = self.penalised_loss(y, theta_y)
        return i_theta

    def scale_norm(self):
        """
        Rescales the norm of the function after the random initialization in order to force it
        inside the interval (0, sigma). This avoids numerical problems in case the starting distribution
        is in a region where the penalization is too large.
        """
        y = self.mu.sample([self.nr_sample, 1])
        theta_y = self.theta(y)
        Lp_norm_theta = torch.pow(torch.mean(torch.pow(torch.abs(theta_y), self.penalised_loss.p)), 1. / self.penalised_loss.p)
        m = torch.distributions.uniform.Uniform(torch.sqrt(self.penalised_loss.h) * self.penalised_loss.penalty.sigma / (5 * Lp_norm_theta),
                        torch.sqrt(self.penalised_loss.h) * self.penalised_loss.penalty.sigma / Lp_norm_theta)
        self.mult_coeff = nn.Parameter(m.sample())



# utility functions for the optimization

def train(model, optimizer):
    """
    Training cycle for a neural network
    :param model: nn module
    :param optimizer: optime module
    :return:
    """
    model.zero_grad()
    output = model()
    output.backward()
    optimizer.step()
    return output
