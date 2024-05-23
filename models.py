# TODO: I think you should also be telling the model t.


import torch as t
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_classes, seq_length, img_c, img_h, img_w, d_hidden=None):
        super().__init__()
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.img_c = img_c
        self.img_h = img_h
        self.img_w = img_w
        self.d_in = num_classes + seq_length + img_c * img_h * img_w  # This is a conditional diffusion model - tell it the label.
        self.d_out = img_c * img_h * img_w
        if d_hidden is None:
            self.d_hidden = self.d_in  # Model still has access to label at hidden dimension.
        else:
            self.d_hidden = d_hidden
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(self.d_in, self.d_hidden)
        self.linear2 = nn.Linear(self.d_hidden, self.d_out)
        self.act = nn.ReLU()

    def forward(self, x):
        # x = self.flatten(x)
        # x = t.cat([y, x], dim=1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x
        # return x.view(-1, self.img_c, self.img_h, self.img_w)

    def generate(self, label, seq_length):
        one_hot_label = F.one_hot(t.tensor(label), num_classes=self.num_classes)
        one_hot_reverse_time_seq = t.eye(seq_length).flipud()
        initial_noise = t.randn(self.img_c * self.img_h * self.img_w)
        X_t = initial_noise
        for i in range(seq_length):
            model_input = t.cat([one_hot_label, one_hot_reverse_time_seq[i], X_t], dim=0).unsqueeze(0)  # (label, X_t)
            model_output = self(model_input)[0]  # Y_t = X_{t-1} - X_t
            X_t_minus_one = model_output + X_t
            X_t = X_t_minus_one

            # model_input = model_input
            # model_input = t.cat([one_hot_label, model_output[0]], dim=0).unsqueeze(0)
        # model_output = model_input[:, self.num_classes:]
        # return model_output.view(1, self.img_h, self.img_w)
        return X_t.view(1, self.img_h, self.img_w)
