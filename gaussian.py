import os
import numpy as np
import torch
from torch import nn


class ThetaSigmoidNetwork(nn.Module):
    def __init__(self, depth, width):
        super(ThetaSigmoidNetwork, self).__init__()
        self.linear_sigm_stack = nn.Sequential(
            nn.Linear(2, width),
            nn.Sigmoid(),
            nn.Linear(width, 2)
        )

    def forward(self, x):
        theta = self.linear_sigm_stack(x)
        return theta


def loss(output, target):

    return

def train(model, x, optimizer, criterion):
    model.zero_grad()
    output = model(x)
    loss = criterion(output, None)
    loss.backward()
    optimizer.step()

    return loss, output


if __name__ == '__main__':

    # np.random.seed(0)
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    theta = ThetaSigmoidNetwork(1, 5).to(device)
    print(theta)

    # let's see the parameters of the model
    print("-----------------------------------------------")
    for name, param in theta.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param} \n")

    print("-----------------------------------------------")
    x = torch.tensor([1., 2.])
    # x = torch.rand(10, 2, device=device)
    print(f"random initial points: {x}")
    directions = theta(x)
    print(directions)

    x = torch.tensor([[1., 2.], [3., 4.]])
    # x = torch.rand(10, 2, device=device)
    print(f"evaluation points: {x}")
    directions = theta(x)
    print(directions)

    criterion = nn.MSELoss()
    EPOCHS = 200
    optm = torch.optim.Adam(theta.parameters(), lr=0.001)