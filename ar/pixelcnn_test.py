from pixelcnn_train import PixelCNN
from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad.nn import datasets
from tinygrad import Tensor, dtypes
import matplotlib.pyplot as plt
import numpy as np


def generate_sample(model, size=(32, 32)):
    """
    Generate a sample image from the PixelCNN model

    Args:
        model: The trained PixelCNN model
        conditioning: Optional conditioning information
        size: Size of the image to generate

    Returns:
        A generated image
    """
    # Initialize with zeros
    image = Tensor.zeros(1, 1, *size).contiguous()

    # Generate the image pixel by pixel
    for i in range(size[0]):
        for j in range(size[1]):
            # Get the current pixel distribution
            pred = model(image)
            # Sample from the distribution (binary image)
            sample = (pred[0, 0, i, j] > 0.5).float()
            # Update the image with the sampled pixel
            image[0, 0, i, j] = sample
            print(j, size[1])
        print("----", i, size[0])
    print("generated image")
    return image


if __name__ == "__main__":
    # Load the trained model
    model = PixelCNN()

    # Load the saved state dictionary
    state_dict = safe_load("model.safetensors")
    load_state_dict(model, state_dict)

    # Load test data
    _, _, X_test, Y_test = datasets.mnist(fashion=True)
    X_test = (
        X_test.cast(dtypes.float32).div(255.0).pad(((0, 0), (0, 0), (2, 2), (2, 2)))
    )

    # Test the model on real data
    samples = Tensor.randint(10, high=X_test.shape[0])
    reconstructions = model(X_test[samples])

    # Generate new samples from scratch
    num_samples = 5
    generated_samples = []

    print("Generating new samples...")
    for i in range(num_samples):
        sample = generate_sample(model)
        generated_samples.append(sample)

    # Visualize reconstructions
    print("Visualizing reconstructions...")
    for i in range(samples.shape[0]):
        plt.figure(figsize=(10, 5))

        # Original image
        plt.subplot(1, 2, 1)
        original = X_test[samples[i]].reshape(32, 32)
        original = original.mul(255).clip(0, 255).cast(dtypes.uint8)
        plt.imshow(original.numpy(), cmap="gray")
        plt.title("Original")

        # Reconstructed image
        plt.subplot(1, 2, 2)
        recon = reconstructions[i].reshape(32, 32)
        recon = recon.mul(255).clip(0, 255).cast(dtypes.uint8)
        plt.imshow(recon.numpy(), cmap="gray")
        plt.title("Reconstructed")

        plt.tight_layout()
        plt.savefig(f"reconstruction_{i}.png")
        plt.close()

    # Visualize generated samples
    print("Visualizing generated samples...")
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        gen_sample = generated_samples[i][0, 0]
        gen_sample = gen_sample.mul(255).clip(0, 255).cast(dtypes.uint8)
        plt.imshow(gen_sample.numpy(), cmap="gray")
        plt.title(f"Sample {i + 1}")

    plt.tight_layout()
    plt.savefig("generated_samples.png")
    plt.close()

    print("Done! Images saved as reconstruction_X.png and generated_samples.png")
