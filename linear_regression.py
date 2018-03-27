from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim, autograd


class LRModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    def train(self, x, y, epochs):
        criterion = nn.MSELoss()
        lr = 0.01
        optimiser = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch += 1
            optimiser.zero_grad()

            outputs = self.forward(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimiser.step()
            print('epoch={}, loss={}'.format(epoch, loss.data[0]))


def main():
    int_dim = 1
    out_dim = 1
    x = np.arange(0, 10).astype(np.float32)
    y = np.random.normal(x).astype(np.float32)
    x = x.reshape(10, 1)

    x = autograd.Variable(torch.from_numpy(x))
    y = autograd.Variable(torch.from_numpy(y))

    print('type of x = {}'.format(type(x)))

    model = LRModel(input_dim=int_dim, output_dim=out_dim)
    model.train(x, y, 20)
    y_predicted = model.forward(x)


if __name__ == '__main__':
    main()









