import torch
import torchsde
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

from src.speed_springs import GGS3DE

FRICTION = 10
TEMP = 1
K = 1
M = 1

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
    pca = PCA(n_components=4)
    u_i_train_pca = torch.tensor(pca.fit_transform(u_i_train_normalized))
    u_i_test_pca = torch.tensor(pca.transform(u_i_test_normalized))


    n_pieces_train = torch.ones_like(u_i_train_pca[0], dtype=int)
    n_pieces_test = torch.ones_like(u_i_test_pca[0], dtype=int)

    batch_size = 1

    state_size_train = torch.prod(n_pieces_train + 1) * 2 * y_i_train.shape[0]
    state_size_test = torch.prod(n_pieces_test + 1) * 2 * y_i_test.shape[0]

    sde_train = GGS3DE(
        n_pieces_train, u_i_train_pca.T, y_i_train.T, friction=FRICTION, temp=TEMP, k=K, M=M
    )
    sde_test = GGS3DE(
        n_pieces_test, u_i_test_pca.T, y_i_test.T, friction=FRICTION, temp=TEMP, k=K, M=M
    )

    ts = torch.linspace(0, MAX_TIME, T_SIZE)
    y0 = torch.rand(size=(batch_size, state_size_train))

    with torch.no_grad():
        ys_gen = torchsde.sdeint(sde_train, y0, ts, method="euler")


    # Save the output
    torch.save(ys_gen, "ys_gen.pt")

    # Calculate the cost
    train_cost = sde_train.cost(ys_gen[:, 0, :]).detach().numpy()/len(u_i_train)
    test_cost = sde_test.cost(ys_gen[:, 0, :]).detach().numpy()/len(u_i_test)

    # Save the cost
    torch.save(train_cost, "train_cost.pt")
    torch.save(test_cost, "test_cost.pt")

def plot():
    ys_gen = torch.load("ys_gen.pt")
    train_cost = torch.load("train_cost.pt")
    test_cost = torch.load("test_cost.pt")

    import matplotlib.pyplot as plt

    plt.plot(ys_gen[:, 0, :].detach().numpy())
    plt.title("Generated Data")
    plt.show()

    plt.plot(train_cost, label="Train Cost")
    plt.plot(test_cost, label="Test Cost")
    plt.legend()
    plt.title("Cost")
    plt.show()

if __name__ == "__main__":
    mnist()
    plot()