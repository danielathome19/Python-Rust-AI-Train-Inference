import torch
import torch.onnx
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split

# Load a suitable dataset
dataset = CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
cifar10_train, cifar10_val = random_split(dataset, [45000, 5000])
train_loader = DataLoader(cifar10_train, batch_size=32)
val_loader = DataLoader(cifar10_val, batch_size=32)

# Grab the first sample from the dataset
_x, _y = next(iter(train_loader))
with open("data/cifar_sample_row.csv", "w") as f:
    f.write(", ".join(f"{x:.1f}" for x in _x[0].flatten().tolist()) + 
            ", " + str(_y[0].item()) + "\n")

# Define and train a CNN model
class CNN(pl.LightningModule):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 1)
        self.pool = torch.nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = torch.nn.Linear(128 * 2 * 2, 256)
        self.fc2 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

model = CNN()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_loader, val_loader)

# Save the trained model in ONNX format
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "models/lt_cnn_model.onnx", input_names=['input'], output_names=['output'])
