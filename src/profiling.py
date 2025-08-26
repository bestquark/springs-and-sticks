import torch
import torchsde
import numpy as np

from src.ff_springs import GroupGS3DE
from torch.utils.data import TensorDataset, DataLoader
import cProfile

import matplotlib.pyplot as plt
import concurrent.futures

import sys
sys.setrecursionlimit(10000)  # or a higher value as needed

def run_main():

    eps = 0.01

    n_points = 10
    u_i_train = torch.rand(n_points, 3)
    y_i_train = (torch.stack([torch.sin(u_i_train[:, 0]*u_i_train[:, 1]*u_i_train[:, 2]), torch.cos(u_i_train[:, 0]*u_i_train[:, 1]*u_i_train[:, 2])]).t() + eps*torch.randn(n_points,2))

    u_i_test = torch.rand(n_points, 3)
    y_i_test = (torch.stack([torch.sin(u_i_test[:,0]*u_i_test[:,1]*u_i_test[:,2]), torch.cos(u_i_test[:,0]*u_i_test[:,1]*u_i_test[:,2])]).t() + eps*torch.randn(n_points,2))

    u_min = torch.min(torch.cat([u_i_train, u_i_test], dim=0), dim=0).values
    u_max = torch.max(torch.cat([u_i_train, u_i_test], dim=0), dim=0).values
    boundaries = (u_min, u_max)


    n_sticks = torch.tensor([2, 2, 2])
    num_labels = y_i_train.shape[1]
    max_features = 2
    fric = 100

    sde = GroupGS3DE(max_features, n_sticks, boundaries, num_labels,
                     friction=fric, temp=0.001, k=1, M=1)

    num_runs, t_size = 1, 2
    num_epochs = 1
    batch_size = 5

    dataset_train = TensorDataset(u_i_train, y_i_train)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    dataset_test = TensorDataset(u_i_test, y_i_test)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    n_batch_train = len(loader_train)
    n_batch_test = len(loader_test)

    train_losses = []
    test_losses = []

    ts = torch.linspace(0, 1, t_size)

    theta0 = None
    with torch.no_grad():
        for epoch in range(num_epochs):
            train_loss = 0
            for i, (u_i, y_i) in enumerate(loader_train):
                sde.update_data(u_i, y_i)
                theta0 = torch.rand(size=(num_runs, sde.state_size)) if theta0 is None else theta0
                loss_sde = sde.loss(theta0.flatten(), u_i, y_i)
                train_loss += loss_sde.item()
                with torch.no_grad():
                    thetas = torchsde.sdeint(sde, theta0, ts, method='euler')
                theta0 = thetas[-1, :]

            train_loss /= n_batch_train
            print(f"Train loss: {train_loss}")
            train_losses.append(train_loss)

            test_loss = 0
            for i, (u_i, y_i) in enumerate(loader_test):
                loss_sde = sde.loss(theta0.flatten(), u_i, y_i)
                test_loss += loss_sde.item()

            test_loss /= n_batch_test
            print(f"Test loss: {test_loss}")
            test_losses.append(test_loss)

            print(f"Epoch {epoch+1}/{num_epochs}")

    # plt.plot(train_losses, label='Train loss')
    # plt.plot(test_losses, label='Test loss')
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    cProfile.run('run_main()', 'profile_stats.prof')