import torch as t


device = "cuda" if t.cuda.is_available() else "cpu"





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
