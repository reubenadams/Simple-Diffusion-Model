import torch as t


device = "cuda" if t.cuda.is_available() else "cpu"


# def get_scales(T):
#     return t.tensor(range(T, -1, -1), device=device) / T
# 
# 
# def get_scaled_images(X0, T):
#     scales = get_scales(T).reshape(T + 1, 1, 1)
#     return X0 * scales


class ScaledSequenceTransform:
    def __init__(self, T):
        self.seq_length = T + 1
        self.scales = t.tensor(range(T, -1, -1), device=device) / T
        self.scales = self.scales.reshape(self.seq_length, 1, 1, 1)

    def __call__(self, X0):
        return X0 * self.scales




# def make_noised_sequence(scaled_images, img_h, img_w, T):
#     noises = t.randn(T, img_h, img_w, device=device) / T
#     scaled_images[1:] += noises
#     return scaled_images


def main():
    T = 3
    X0 = t.arange(4).reshape(2, 2)
    # scaled_images = get_scaled_images(X0, T)
    # print(f"Scaled images shape: {scaled_images.shape}")
    # sequence = make_sequence(scaled_images, X0.shape[0], X0.shape[1], T)
    # print(sequence)
    trans = ScaledSequenceTransform(T=T)
    X_seq = trans(X0)
    print(X_seq)



if __name__ == "__main__":
    main()
