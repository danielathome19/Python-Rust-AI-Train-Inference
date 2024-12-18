import torch
import torch.onnx
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

# Load a suitable dataset
dataset = MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])
train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

# Grab the first sample from the dataset
_x, _y = next(iter(train_loader))
with open("data/mnist_sample_row.csv", "w") as f:
    f.write(", ".join(f"{x:.1f}" for x in _x[0].flatten().tolist()) + 
            ", " + str(_y[0].item()) + "\n")

# Define and train an FNN model
class FNN(pl.LightningModule):
    def __init__(self):
        super(FNN, self).__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

model = FNN()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, train_loader, val_loader)

# Save the trained model in ONNX format
dummy_input = torch.randn(1, 28 * 28)
torch.onnx.export(model, dummy_input, "models/lt_fnn_model.onnx", input_names=['input'], output_names=['output'])

# Print the first sample from the validation set
sample = next(iter(val_loader))
x, y = sample
print(x)

# Print the model's prediction for the first sample
y_hat = model(x)
print(y_hat)
