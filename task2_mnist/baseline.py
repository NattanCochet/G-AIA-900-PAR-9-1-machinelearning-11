"""PyTorch port of the course's Keras MNIST baseline (main_learn_net.py):
Conv2d(28, 3x3) -> MaxPool(2x2) -> Flatten -> Linear(128) + ReLU -> Dropout(0.2)
-> Linear(10), Adam, 10 epochs, batch size 32. Demonstrates the common
utilities (seed, phase timer, CSV logger, figure style); this is a baseline
run, not yet the full "time to 0.97" protocol (task 2's own tickets).
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common import PhaseTimer, log_result, new_figure, set_seed, set_style

SEED = 0
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "task2_mnist/data"
RESULTS_CSV = "results/task2_mnist.csv"
FIGURES_DIR = "results"


class CourseBaselineCNN(nn.Module):
    """Same architecture as the course's Keras model. Keras's Conv2D here has
    no activation, so this stays logit-only until the final layer; the
    softmax from the Keras model is folded into nn.CrossEntropyLoss."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=28, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(28 * 13 * 13, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv(x))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def load_data() -> tuple[DataLoader, DataLoader]:
    to_tensor = transforms.ToTensor()  # scales pixels to [0, 1], like /255
    train_set = datasets.MNIST(
        DATA_DIR, train=True, download=True, transform=to_tensor
    )
    test_set = datasets.MNIST(
        DATA_DIR, train=False, download=True, transform=to_tensor
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    return train_loader, test_loader


def train(model: nn.Module, train_loader: DataLoader) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    epoch_losses = []
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        epoch_losses.append(total_loss / len(train_loader.dataset))
        print(f"epoch {epoch + 1}/{EPOCHS}: train loss={epoch_losses[-1]:.4f}")
    return epoch_losses


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return correct / total


def plot_training_curve(epoch_losses: list[float]) -> None:
    set_style()
    fig, ax = new_figure(
        xlabel="epoch", ylabel="train loss", title="MNIST baseline training curve"
    )
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
    fig.savefig(f"{FIGURES_DIR}/task2_mnist_baseline_loss.png", bbox_inches="tight")


def main() -> None:
    set_seed(SEED)

    timer = PhaseTimer()
    with timer.phase("load"):
        train_loader, test_loader = load_data()

    model = CourseBaselineCNN()
    with timer.phase("train"):
        epoch_losses = train(model, train_loader)

    with timer.phase("eval"):
        test_accuracy = evaluate(model, test_loader)

    print(f"test accuracy: {test_accuracy:.4f}")
    plot_training_curve(epoch_losses)

    log_result(
        RESULTS_CSV,
        task="task2_mnist",
        seed=SEED,
        split="test",
        metric="accuracy",
        score=test_accuracy,
        duration_seconds=timer.total("train"),
        config={
            "model": "course_baseline_cnn",
            "optimizer": "adam",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "load_seconds": timer.times["load"],
            "eval_seconds": timer.times["eval"],
        },
    )


if __name__ == "__main__":
    main()
