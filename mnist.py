"""
Raw python example.
Based on https://raw.githubusercontent.com/pytorch/examples/master/mnist/main.py.
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

_BATCH_SIZE = 64
_EPOCHS = 1000
_GAMMA = 0.7
_LEARNING_RATE = 1.0
_LOG_INTERVAL = 10
_VAL_BATCH_SIZE = 1000


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_data_loaders():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=_BATCH_SIZE,
        shuffle=True,
        num_workers=1,
    )
    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=_VAL_BATCH_SIZE,
        num_workers=1,
    )
    return train_loader, val_loader


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % _LOG_INTERVAL == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)

    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {:.0f}%\n".format(
            val_loss, 100.0 * correct / len(val_loader.dataset),
        )
    )


def main():
    device = torch.device("cuda")

    train_loader, val_loader = get_data_loaders()
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=_LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=1, gamma=_GAMMA)

    for epoch in range(1, _EPOCHS + 1):
        start_time = time.time()
        train(model, device, train_loader, optimizer, epoch)
        end_time = time.time()
        print(f"Epoch {epoch} took {int(end_time - start_time)} seconds")

        validate(model, device, val_loader)
        scheduler.step()

    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
