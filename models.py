import torch as t
from torch import nn


class MLP(nn.Module):
    def __init__(self, num_labels, img_c, img_h, img_w, seq_length, d_hidden=None):
        super().__init__()
        self.num_labels = num_labels
        self.img_c = img_c
        self.img_h = img_h
        self.img_w = img_w
        self.seq_length = seq_length
        self.d_in = num_labels + img_c * img_h * img_w  # This is a conditional diffusion model - tell it the label.
        self.d_out = img_c * img_h * img_w
        if d_hidden is None:
            self.d_hidden = self.d_in  # Model still has access to label at hidden dimension.
        else:
            self.d_hidden = d_hidden
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(self.d_in, self.d_hidden)
        self.linear2 = nn.Linear(self.d_hidden, self.d_out)
        self.act = nn.ReLU()

    def forward(self, x, y):
        x = self.flatten(x)
        x = t.cat([y, x], dim=1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x.view(-1, self.img_c, self.img_h, self.img_w)
