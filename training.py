import torch as t
from models import MLP
import torch.nn as nn


device = "cuda" if t.cuda.is_available() else "cpu"


def get_noised_sequence(batch_scaled_images):
    batch_size, seq_length, img_c, img_h, img_w = batch_scaled_images.shape
    T = seq_length - 1
    noises = t.randn(batch_size, T, img_c, img_h, img_w, device=device) / T
    batch_scaled_images[:, 1:] += noises
    return batch_scaled_images


img_c = 1
img_h = img_w = 28

model = MLP(num_classes=10, img_c=img_c, img_h=img_h, img_w=img_w, seq_length=10)


def train(model, train_seqs_loader, lr=1e-3):
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    for batch_scaled_images in train_seqs_loader:
        batch_scaled_images.to(device)
        batch_noised_images = get_noised_sequence(batch_scaled_images)  # (X0, ..., XT)
        model_input = batch_noised_images[1:]
        model_target = batch_noised_images[:-1] - model_input

        optimizer.zero_grad()
        model_output = model(model_input)
        loss = loss_fn(model_output, model_target)
        loss.backward()
        optimizer.step()
        break
