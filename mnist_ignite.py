"""
Pytorch Ignite example.
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from ignite.engine import create_supervised_trainer, Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

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


def main():
    device = torch.device("cuda")

    train_loader, val_loader = get_data_loaders()
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=_LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=1, gamma=_GAMMA)

    trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device)
    evaluator = create_supervised_evaluator(
        model, metrics={"accuracy": Accuracy(), "nll": Loss(F.nll_loss)}, device=device
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=_LOG_INTERVAL))
    def log_training_loss(engine):
        loss = engine.state.output
        epoch = engine.state.epoch
        iteration = engine.state.iteration % len(train_loader)
        print(
            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                iteration * _BATCH_SIZE,
                len(train_loader.dataset),
                100.0 * iteration / len(train_loader),
                loss,
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate_model(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        val_loss = metrics["nll"]
        print(
            "\nValidation set: Average loss: {:.4f}, Accuracy: {:.0f}%\n".format(
                val_loss, 100.0 * avg_accuracy,
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def scheduler_step(engine):
        scheduler.step()

    @trainer.on(Events.EPOCH_STARTED)
    def start_epoch_timer(engine):
        engine.state.start_time = time.time()

    @trainer.on(Events.EPOCH_COMPLETED)
    def end_epoch_timer(engine):
        start_time = engine.state.start_time
        end_time = time.time()
        epoch = engine.state.epoch
        print(f"Epoch {epoch} took {int(end_time - start_time)} seconds")

    trainer.run(train_loader, max_epochs=_EPOCHS)

    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
