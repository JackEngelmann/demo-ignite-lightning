"""
Small example of PyTorch Lightning on MNIST data.
"""
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

_BATCH_SIZE = 64
_EPOCHS = 1000
_GAMMA = 0.7
_LEARNING_RATE = 1.0
_LOG_INTERVAL = 10
_VAL_BATCH_SIZE = 1000


class Net(nn.Module):
    def __init__(self):
        super().__init__()
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


class LightningNet(LightningModule):
    def __init__(self):
        super().__init__()
        self.net = Net()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.nll_loss(output, y)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters(), lr=_LEARNING_RATE)
        scheduler = StepLR(optimizer, step_size=1, gamma=_GAMMA)
        return [optimizer], [scheduler]

    def prepare_data(self):
        # Lightning throws an error when there is no prepare_data method.
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            datasets.MNIST(
                "../data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=_BATCH_SIZE,
            num_workers=1,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
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

    def validation_step(self, batch, batch_idx):
        x, target = batch
        output = self(x)
        loss = F.nll_loss(output, target, reduction="sum")
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum()
        return {"val_loss": loss, "correct": correct}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(tuple(x["val_loss"] for x in outputs)).mean()
        correct = torch.stack(tuple(x["correct"] for x in outputs)).sum()
        return {"avg_loss": avg_loss, "correct": correct}


class EpochTimer(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None

    def on_epoch_start(self, trainer, module):
        self.start_time = time.time()

    def on_epoch_end(self, trainer, module):
        epoch = trainer.current_epoch
        start_time = self.start_time
        end_time = time.time()
        print(f"Epoch {epoch} took {int(end_time - start_time)} seconds")


def main():
    model = LightningNet()

    trainer = Trainer(gpus=1, max_epochs=_EPOCHS, callbacks=[EpochTimer()],)
    trainer.fit(model)

    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
