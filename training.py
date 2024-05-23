from models import MLP
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

img_c = 1
img_h = img_w = 28


model = MLP(10, img_c, img_h, img_w, seq_length=10)
loss_fn = nn.MSELoss()


for x, y in train_loader:
    y = F.one_hot(y, num_classes=10)
    y_hat = model(x, y)
    loss = loss_fn(y_hat, x)
    print(loss)
    break
