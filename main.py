import torch as t
from data_preparation import get_data_loaders
from models import MLP
from training import train
import matplotlib.pyplot as plt


device = "cuda" if t.cuda.is_available() else "cpu"


img_c = 1
img_h = img_w = 28
T = 15
seq_length = T + 1
model = MLP(num_classes=10, seq_length=seq_length, img_c=img_c, img_h=img_h, img_w=img_w)
train_loader, test_loader = get_data_loaders(T=T)
train(model, train_loader, seq_length, num_classes=10)

for i in range(10):
    gen_image = model.generate(label=i, seq_length=seq_length)
    plt.imshow(gen_image.detach().cpu().numpy()[0])
    plt.show()

