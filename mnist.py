import numpy as np

import torch
import torchsde
from torchvision import datasets, transforms

import sys

sys.path.append("./src")

from sklearn.decomposition import PCA
from springs_sdes import GGS3DE


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]
)


def main(func, ndata_train=100):
    trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=True
    )

    u_i_train, y_train = next(iter(trainloader))
    y_i_train = torch.nn.functional.one_hot(y_train, num_classes=10).float()

    u_i_train = u_i_train.numpy()
    y_i_train = y_i_train.numpy()

    # Get 10 PCA components of u_i_train
    u_i_train_centered = u_i_train - u_i_train.mean(axis=0)
    u_i_train_normalized = u_i_train_centered / (u_i_train_centered.std(axis=0) + 1e-10)

    pca = PCA(n_components=10)
    u_i_train = pca.fit_transform(u_i_train_normalized)

    u_i_train = u_i_train[:ndata_train, :]
    y_i_train = y_i_train[:ndata_train, :]

    # Train the springs and sticks model
    n_pieces = 2 * np.ones_like(u_i_train[0], dtype=int)
    batch_size, state_size, t_size = 2, np.prod(n_pieces) * 2 * y_i_train.shape[0], 200
    sde = GGS3DE(
        n_pieces, u_i_train.T, y_i_train.T, friction=5, temp=1, k=1e-15, M=1e-15
    )
    ts = torch.linspace(0, 20, t_size)

    def train():
        y0 = torch.rand(size=(batch_size, state_size))
        with torch.no_grad():
            ys_gen = torchsde.sdeint(
                sde, y0, ts, method="euler"
            )  # (t_size, batch_size, state_size)
        np.save("ys_gen.npy", ys_gen)

    def test():
        testset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset), shuffle=True
        )

        u_i_test, y_test = next(iter(testloader))
        y_i_test = torch.nn.functional.one_hot(y_test, num_classes=10).float()

        ys_gen = np.load("ys_gen.npy")
        # TODO: get error for testset

    if func == "train":
        train()
    elif func == "test":
        test()
    elif func == "both":
        train()
        test()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--func", choices=["train", "test", "both"])
    parser.add_argument("--ndata_train", type=int, default=100)
    args = parser.parse_args()

    main(args.func, args.ndata_train)
