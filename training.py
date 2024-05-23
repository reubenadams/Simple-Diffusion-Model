import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


device = "cuda" if t.cuda.is_available() else "cpu"


def get_noised_image_seqs(scaled_image_seqs):
    noised_image_seqs = scaled_image_seqs.clone()   # (batch_size, seq_length, img_c, img_h, img_w)
    batch_size, seq_length, img_c, img_h, img_w = scaled_image_seqs.shape
    T = seq_length - 1
    noises = t.randn(batch_size, T, img_c, img_h, img_w, device=device) / T
    noised_image_seqs[:, 1:] += noises
    return noised_image_seqs                        # (batch_size, seq_length, img_c, img_h, img_w)


def get_one_hot_label_seqs(one_hot_labels, seq_length):
    return one_hot_labels.unsqueeze(1).repeat(1, seq_length, 1)  # (batch_size, seq_length, num_classes)


def get_one_hot_time_seqs(batch_size, seq_length):
    one_hot_time_seqs = t.eye(seq_length).unsqueeze(0).tile(dims=(batch_size, 1, 1))  # (batch_size, seq_length, seq_length)
    # time_stamps = t.arange(seq_length)
    # time_stamps = time_stamps.unsqueeze(0).tile(dims=(batch_size, 1))  # (batch_size, seq_length)
    return one_hot_time_seqs
    # return t.flatten(one_hot_time_stamps, start_dim=0, end_dim=1)


def get_training_samples(scaled_image_seqs, one_hot_labels, seq_length, num_classes):
    noised_image_seqs = get_noised_image_seqs(scaled_image_seqs)   # (batch_size, seq_length, img_c, img_h, img_w)
    noised_image_seqs = t.flatten(noised_image_seqs, start_dim=2)  # (batch_size, seq_length, img_c * img_h * img_w)
    one_hot_label_seqs = get_one_hot_label_seqs(one_hot_labels, seq_length)  # (batch_size, seq_length, num_classes)
    one_hot_time_seqs = get_one_hot_time_seqs(batch_size=scaled_image_seqs.shape[0], seq_length=seq_length)
    data_seqs = t.cat([one_hot_label_seqs, one_hot_time_seqs, noised_image_seqs], dim=2)  #   (batch_size, seq_length, num_classes + seq_length + img_c * img_h * img_w)
    model_input_seqs = data_seqs[:, 1:, :]      # (X1, ..., XT)            (batch_size, T, num_classes + img_c * img_h * img_w)
    model_target_seqs = data_seqs[:, :-1, :]    # (X0, ..., X_{T-1})       (batch_size, T, num_classes + img_c * img_h * img_w)
    model_inputs = t.flatten(model_input_seqs, start_dim=0, end_dim=1)   # (batch_size * T, num_classes + img_c * img_h * img_w)
    model_targets = t.flatten(model_target_seqs, start_dim=0, end_dim=1) # (batch_size * T, num_classes + img_c * img_h * img_w)
    model_targets = model_targets[:, num_classes + seq_length:]
    return model_inputs, model_targets


def train(model, train_seqs_loader, seq_length, num_classes, lr=1e-2, epochs=1):
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for i in range(epochs):
        print("Epoch:", i)
        for j, (scaled_image_seqs, one_hot_labels) in enumerate(train_seqs_loader):
            if j % 100 == 0:
                print("\tBatch:", j)
            scaled_image_seqs.to(device)  # (batch_size, seq_length, img_c, img_h, img_w)
            one_hot_labels.to(device)  # (batch_size, num_classes)

            model_inputs, model_targets = get_training_samples(scaled_image_seqs, one_hot_labels, seq_length, num_classes)


            # model_input_x = get_noised_image_seqs(scaled_image_seqs)  # (X0, ..., XT)  # (batch_size * seq_length, img_c, img_h, img_w)
            # model_input_x = batch_noised_images[:, 1:]  # (batch_size, seq_length - 1, img_c, img_h, img_w)

            # model_input_y = get_expanded_batch_labels(batch_one_hot_labels, seq_length)  # (batch_size, num_classes)
            # model_target = batch_noised_images[:, :-1] - model_input_x

            optimizer.zero_grad()
            # model_output = model(model_input_x, model_input_y)
            model_outputs = model(model_inputs)
            loss = loss_fn(model_outputs, model_targets)
            loss.backward()
            optimizer.step()

            losses.append(loss)
    plt.plot([loss.detach().cpu() for loss in losses])
    plt.yscale("log")
    plt.show()
