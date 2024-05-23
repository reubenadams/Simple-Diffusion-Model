# TODO: Break up the sequences?


import torch as t
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


device = "cuda" if t.cuda.is_available() else "cpu"


class ScaledSequenceTransform:
    def __init__(self, T):
        self.seq_length = T + 1
        self.scales = t.tensor(range(T, -1, -1), device=device) / T
        self.scales = self.scales.reshape(self.seq_length, 1, 1, 1)

    def __call__(self, X0):
        return X0 * self.scales


class OneHotTargetTransform:
    def __init__(self, num_classes):
        self.num_classes = num_classes


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ScaledSequenceTransform(T=32),
])


train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform,
                               target_transform=lambda y: F.one_hot(t.tensor(y), num_classes=10))
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform,
                              target_transform=lambda y: F.one_hot(t.tensor(y), num_classes=10))
# test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)



