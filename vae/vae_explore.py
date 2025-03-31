from vae_train import VAEModel, EncoderModel, DecoderModel
from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad.nn import datasets
from tinygrad import Tensor, dtypes
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

if __name__ == "__main__":
    encoder = EncoderModel()
    decoder = DecoderModel()

    model = VAEModel(encoder, decoder)

    # and load it back in
    state_dict = safe_load("model.safetensors")
    load_state_dict(model, state_dict)

    X_train, Y_train, X_test, Y_test = datasets.mnist(fashion=True)

    # First, let's create a function to encode all test data
    def encode_batch(x):
        z_sample, z_mean, z_log_var = encoder(x)
        return z_mean, z_log_var, z_sample

    # Encode all test data to get latent vectors
    batch_size = 500
    latent_vectors = []
    labels = []

    for i in range(0, len(X_test), batch_size):
        end = min(i + batch_size, len(X_test))
        batch = X_test[i:end]
        # Reshape batch to expected format (batch_size, channels, height, width)
        batch_reshaped = batch.reshape(-1, 1, 28, 28)
        z_mean, _, _ = encode_batch(batch_reshaped)
        latent_vectors.append(z_mean.numpy())
        labels.append(Y_test[i:end].numpy())

    # Concatenate all batches
    latent_vectors = np.vstack(latent_vectors)
    labels = np.concatenate(labels)

    # Create a scatter plot of the latent space
    plt.figure(figsize=(12, 10))
    plt.subplot(1, 1, 1)

    # Color map for the 10 classes in Fashion-MNIST
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    # Class names for Fashion-MNIST
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # Plot each class with a different color
    for i in range(10):
        idx = np.where(labels == i)[0]
        plt.scatter(
            latent_vectors[idx, 0],
            latent_vectors[idx, 1],
            c=colors[i],
            label=class_names[i],
            alpha=0.6,
            s=10,
        )

    plt.legend()
    plt.title("Latent Space Visualization")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.tight_layout()
    plt.savefig("latent_space.png")
    plt.close()

    # Now, let's create a grid of reconstructed images from the latent space
    n = 15  # Grid size
    figure = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(n, n)
    grid_x = np.linspace(-3.0, 3.0, n)
    grid_y = np.linspace(-3.0, 3.0, n)

    # Function to generate images from latent vectors
    def generate_image(z):
        return decoder(z)

    for i, xi in enumerate(grid_x):
        for j, yi in enumerate(grid_y):
            z_sample = Tensor([[xi, yi]])
            x_decoded = generate_image(z_sample)
            # The decoder produces 32x32 images, but we want to display them properly
            digit = x_decoded.reshape(1, 32, 32).numpy()[0]

            ax = plt.subplot(gs[i, j])
            ax.imshow(digit, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("latent_grid.png")
    plt.close()

    print("Visualization complete! Check the generated PNG files.")
