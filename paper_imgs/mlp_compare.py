#!/usr/bin/env python
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchsde
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from src.ff_springs import GroupGS3DE, GS3DE

# LaTeX configuration for matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
})
plt.rc('text.latex', preamble=r'\usepackage{amsmath, upgreek}')

def train_function(func_name, func, num_epochs=10, batch_size=16, num_runs=1, t_size=2, fric=100, eps=0.01):
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate training data: 160 points in 2 dims
    N_train = 160
    u_i_train = torch.rand(N_train, 2)
    y_i_train = func(u_i_train) + eps * torch.randn(N_train)
    y_i_train = y_i_train.unsqueeze(1) 
    
    # Generate test data: 160 points in 2 dims
    N_test = 160
    u_i_test = torch.rand(N_test, 2)
    y_i_test = func(u_i_test) + eps * torch.randn(N_test)
    y_i_test = y_i_test.unsqueeze(1)
    
    # Determine boundaries from combined data for SDE initialization
    u_cat = torch.cat([u_i_train, u_i_test], dim=0)
    u_min = torch.min(u_cat, dim=0).values
    u_max = torch.max(u_cat, dim=0).values
    boundaries = (u_min, u_max)
    
    # Create DataLoaders
    dataset_train = TensorDataset(u_i_train, y_i_train)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    dataset_test = TensorDataset(u_i_test, y_i_test)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    # SDE model parameters: mapping ℝ² → ℝ, so use max_features=1.
    # For a 2D input, we use n_sticks of size [2,2] (hidden dimension = 4)
    n_sticks = torch.tensor([2, 2])
    num_labels = 1

    sde = GS3DE(n_sticks, boundaries, num_labels,
                     friction=fric, temp=0.001, k=1, M=1)
    
    # Define a simple MLP: 2 → hidden_dim → 1
    input_dim = 2
    hidden_dim = int(torch.prod(n_sticks).item())  # here, 2*2 = 4
    mlp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_labels)
    )
    
    # (Optional) Initialize biases randomly and freeze them:
    # with torch.no_grad():
    #     for layer in mlp:
    #         if isinstance(layer, nn.Linear) and layer.bias is not None:
    #             layer.bias.copy_(torch.randn_like(layer.bias))
    #             layer.bias.requires_grad = False

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(mlp.parameters(), lr=0.1)
    
    # Time steps for SDE integration
    ts = torch.linspace(0, 1, t_size)
    
    # Containers for loss histories
    sde_train_losses = []
    mlp_train_losses = []
    sde_test_losses = []
    mlp_test_losses = []
    
    theta0 = None  # initial condition for SDE integration
    
    for epoch in range(num_epochs):
        sde_epoch_loss = 0.0
        mlp_epoch_loss = 0.0
        n_batch_train = len(loader_train)
        
        for u_batch, y_batch in loader_train:
            # Update SDE with current batch (note transpose: shape becomes [features, batch_size])
            sde.update_data(u_batch, y_batch)
            theta0 = torch.rand(size=(num_runs, sde.state_size))
            loss_sde = sde.loss(theta0.flatten(), u_batch, y_batch)
            sde_epoch_loss += loss_sde.item()
            with torch.no_grad():
                thetas = torchsde.sdeint(sde, theta0, ts, method='euler')
            theta0 = thetas[-1, :]
            
            optimizer.zero_grad()
            mlp_out = mlp(u_batch)
            loss_mlp = criterion(mlp_out, y_batch)
            loss_mlp.backward()
            optimizer.step()
            mlp_epoch_loss += loss_mlp.item()
        
        sde_epoch_loss /= n_batch_train
        mlp_epoch_loss /= n_batch_train
        sde_train_losses.append(sde_epoch_loss)
        mlp_train_losses.append(mlp_epoch_loss)
        
        # Evaluation on test data
        sde_test_loss = 0.0
        mlp_test_loss = 0.0
        n_batch_test = len(loader_test)
        with torch.no_grad():
            for u_batch, y_batch in loader_test:
                sde_test_loss += sde.loss(theta0.flatten(), u_batch, y_batch).item()
                mlp_out = mlp(u_batch)
                mlp_test_loss += criterion(mlp_out, y_batch).item()
        sde_test_loss /= n_batch_test
        mlp_test_loss /= n_batch_test
        sde_test_losses.append(sde_test_loss)
        mlp_test_losses.append(mlp_test_loss)
        
        print(f"Function: {func_name}, Epoch {epoch+1}/{num_epochs} -- "
              f"SDE Train: {sde_epoch_loss:.4f}, MLP Train: {mlp_epoch_loss:.4f}, "
              f"SDE Test: {sde_test_loss:.4f}, MLP Test: {mlp_test_loss:.4f}")
    
    return {
        "sde_train_losses": sde_train_losses,
        "mlp_train_losses": mlp_train_losses,
        "sde_test_losses": sde_test_losses,
        "mlp_test_losses": mlp_test_losses
    }

def main():
    # Define eight functions from ℝ² to ℝ
    functions = {
        "\sin(x)+\cos(y)": lambda u: torch.sin(u[:,0]) + torch.cos(u[:,1]),
        "x \cdot y": lambda u: u[:,0] * u[:,1],
        "x^2 - y^2": lambda u: u[:,0]**2 - u[:,1]**2,
        "\exp(-x^2-y^2)": lambda u: torch.exp(- (u[:,0]**2 + u[:,1]**2)),
        "\log(1+|x|)-\log(1+|y|)": lambda u: torch.log1p(torch.abs(u[:,0])) - torch.log1p(torch.abs(u[:,1])),
        "\tanh(x \cdot y)": lambda u: torch.tanh(u[:,0] * u[:,1]),
        "\sin(x \cdot y)": lambda u: torch.sin(u[:,0] * u[:,1]),
        "\cos(x)+\sin(y)": lambda u: torch.cos(u[:,0]) + torch.sin(u[:,1]),
    }
    
    results = {}
    num_epochs = 10  # Adjust as desired
    
    for func_name, func in functions.items():
        print(f"\n--- Training on function: {func_name} ---")
        results[func_name] = train_function(func_name, func, num_epochs=num_epochs, batch_size=16)
    
    # Save results to JSON
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Load the results back from JSON
    with open("results.json", "r") as f:
        loaded_results = json.load(f)
    
    # Create a 4x2 grid plot of the loss curves
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    axes = axes.flatten()
    epochs = list(range(1, num_epochs + 1))
    
    for ax, (func_name, data) in zip(axes, loaded_results.items()):
        ax.plot(epochs, data["sde_train_losses"], 'o-', label='SDE Train')
        ax.plot(epochs, data["sde_test_losses"], 'o--', label='SDE Test')
        ax.plot(epochs, data["mlp_train_losses"], '^-', label='MLP Train')
        ax.plot(epochs, data["mlp_test_losses"], '^--', label='MLP Test')
        ax.set_title(rf"${func_name}$", fontsize=12)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig("grid_plot.pdf")
    plt.show()

if __name__ == "__main__":
    main()