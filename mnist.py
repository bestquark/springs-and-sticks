import torch
import torchsde
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

from src.speed_springs import GGS3DE

FRICTION = 50
TEMP = 1
K = 1
M = 1

BATCH_SIZE = 20

PCA_COMPONENTS = 4

MAX_TIME=20
T_SIZE=100

def mnist():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]
    )

    trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=True
    )

    testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=True
    )

    u_i_train, y_train = next(iter(trainloader))
    u_i_test, y_test = next(iter(testloader))

    y_i_train = torch.nn.functional.one_hot(y_train, num_classes=10)
    y_i_test = torch.nn.functional.one_hot(y_test, num_classes=10)

    print("Preprocessing data...")

    u_i_train_centered = u_i_train - u_i_train.mean(axis=0)
    u_i_train_normalized = u_i_train_centered / (u_i_train_centered.std(axis=0) + 1e-10)

    u_i_test_centered = u_i_test - u_i_train.mean(axis=0)
    u_i_test_normalized = u_i_test_centered / (u_i_train.std(axis=0) + 1e-10)

    # Get 4 PCA components of u_i_train
    pca = PCA(n_components=PCA_COMPONENTS)
    u_i_train_pca = torch.tensor(pca.fit_transform(u_i_train_normalized))
    u_i_test_pca = torch.tensor(pca.transform(u_i_test_normalized))


    n_pieces_train = torch.ones_like(u_i_train_pca[0], dtype=int)
    n_pieces_test = torch.ones_like(u_i_test_pca[0], dtype=int)    

    state_size_train = torch.prod(n_pieces_train + 1) * 2 * y_i_train.shape[0]
    state_size_test = torch.prod(n_pieces_test + 1) * 2 * y_i_test.shape[0]

    print("Creating train and test models...")

    sde_train = GGS3DE(
        n_pieces_train, u_i_train_pca.T, y_i_train.T, friction=FRICTION, temp=TEMP, k=K, M=M
    )
    sde_test = GGS3DE(
        n_pieces_test, u_i_test_pca.T, y_i_test.T, friction=FRICTION, temp=TEMP, k=K, M=M
    )

    print("Saving models...")

    idx = f"{FRICTION}_{TEMP}_{K}_{M}_{PCA_COMPONENTS}_{MAX_TIME}_{T_SIZE}"

    # Save the models with pickle
    import pickle

    with open(f"runs/sde_train_{idx}.pkl", "wb") as f:
        pickle.dump(sde_train, f)

    with open(f"runs/sde_test_{idx}.pkl", "wb") as f:
        pickle.dump(sde_test, f)

    print("Time evolving the models...")

    ts = torch.linspace(0, MAX_TIME, T_SIZE)
    y0 = torch.rand(size=(BATCH_SIZE, state_size_train))

    with torch.no_grad():
        ys_gen = torchsde.sdeint(sde_train, y0, ts, method="euler")


    # Save the output
    torch.save(ys_gen, f"ys_gen_{idx}.pt")

    # Calculate the cost
    train_cost = sde_train.cost(ys_gen[:, 0, :]).detach().numpy()/len(u_i_train)
    test_cost = sde_test.cost(ys_gen[:, 0, :]).detach().numpy()/len(u_i_test)

    # Save the cost
    torch.save(train_cost, f"train_cost_{idx}.pt")
    torch.save(test_cost, f"test_cost_{idx}.pt")

    return idx

def plot(idx):
    ys_gen = torch.load(f"ys_gen_{idx}.pt")
    train_cost = torch.load(f"train_cost_{idx}.pt")
    test_cost = torch.load(f"test_cost_{idx}.pt")

    friction, temp, k, m, pca_components, max_time, t_size = idx.split("_")

    import matplotlib.pyplot as plt

    plt.title(f"Friction: {friction}, Temp: {temp}, K: {k}, M: {m}", fontsize=20)
    plt.plot(train_cost, label="Train Cost", color='blue')
    plt.plot(test_cost, label="Test Cost", color='red')
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Cost", fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(f"figs/cost_{idx}.pdf")

if __name__ == "__main__":
    idx = mnist()
    plot(idx)