from pixelcnn_train import PixelCNN, PIXEL_LEVELS
from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad.nn import datasets
from tinygrad import Tensor, dtypes
import matplotlib.pyplot as plt
import numpy as np
import os


class PixelCNNGenerator:
    def __init__(self, model: PixelCNN, num_img=10, input_shape=(1, 1, 32, 32)):
        """
        Initialize a PixelCNN image generator

        Args:
            model: The trained PixelCNN model
            num_img: Number of images to generate (default=10)
            input_shape: Shape of the input images (default=(1, 1, 32, 32))
        """
        self.model: PixelCNN = model
        self.num_img = num_img
        self.input_shape = input_shape

    def sample_from(self, probs, temperature=1.0):
        """
        Sample from a categorical distribution with temperature

        Args:
            probs: Probabilities for each category
            temperature: Temperature parameter for sampling (higher = more random)

        Returns:
            Sampled category index
        """
        if temperature == 0:
            # If temperature is 0, just return the most likely class
            return np.argmax(probs)

        # Apply temperature to the probabilities
        probs = np.exp(np.log(probs + 1e-10) / temperature)
        probs = probs / np.sum(probs)

        # Sample from the distribution
        return np.random.choice(len(probs), p=probs)

    def generate(self, temperature=0.7):
        """
        Generate multiple images using PixelCNN with temperature sampling

        Args:
            temperature: Temperature for sampling (higher = more random)

        Returns:
            Generated images as numpy array
        """
        # Create empty array for generated images
        _, channels, height, width = self.input_shape
        generated_images = np.zeros((self.num_img, channels, height, width))

        # Generate each image pixel by pixel
        for img_idx in range(self.num_img):
            # Initialize with zeros
            image = Tensor.zeros(1, channels, height, width).contiguous()

            # Generate the image pixel by pixel
            for row in range(height):
                for col in range(width):
                    # Get the current pixel distribution
                    logits = self.model(image)
                    # Convert to numpy for sampling
                    probs = logits.softmax(axis=1).permute(0, 2, 3, 1).numpy()
                    probs = probs[0, row, col, :]

                    # Sample from the distribution and normalize
                    sampled_value = self.sample_from(probs, temperature)
                    normalized_value = sampled_value / PIXEL_LEVELS
                    print(normalized_value)
                    # Update the image with the sampled pixel
                    image[0, 0, row, col] = normalized_value

                if (row + 1) % 8 == 0:  # Print progress every 8 rows
                    print(
                        f"Image {img_idx + 1}/{self.num_img}: Generated row {row + 1}/{height}"
                    )

            # Store the generated image
            generated_images[img_idx] = image.numpy()
            print(f"Completed generating image {img_idx + 1}/{self.num_img}")

        return generated_images


def visualize_generated_images(images, save_path="generated_samples.png"):
    """
    Visualize the generated images in a grid

    Args:
        images: Generated images in numpy format
        save_path: Path to save the visualization (default='generated_samples.png')
    """
    # Define the grid size
    num_images = images.shape[0]
    cols = min(5, num_images)
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(cols * 2, rows * 2))

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)

        # Reshape and scale the image
        img = images[i, 0]  # Take the first channel
        print(img)
        img = img * 255

        # Display the image
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.title(f"Sample {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.show()


def generate_random_images(num_images=10, temperature=0.8):
    """
    Main function to generate random images using a trained PixelCNN model

    Args:
        num_images: Number of images to generate
        temperature: Temperature for sampling (higher = more random)
    """
    # Path to the model file
    model_path = "model.safetensors"

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("You need to train the model first by running pixelcnn_train.py.")
        print("Command: python ar/pixelcnn_train.py")
        return

    print(f"Loading model from {model_path}...")

    # Load the trained model
    model = PixelCNN()

    try:
        # Load the saved state dictionary
        state_dict = safe_load(model_path)
        load_state_dict(model, state_dict)

        # Create a generator and generate images
        generator = PixelCNNGenerator(model, num_img=num_images)
        print(f"Generating {num_images} images with temperature {temperature}...")
        generated_images = generator.generate(temperature=temperature)

        # Visualize the generated images
        print("Visualizing results...")
        visualize_generated_images(generated_images)

    except Exception as e:
        print(f"Error loading or using model: {e}")
        print("Make sure you have trained the model correctly.")


def generate_and_visualize_fake_images(num_images=10):
    """
    Generate fake random images for demonstration when no model is available

    Args:
        num_images: Number of fake images to generate
    """
    print("Generating fake random images for demonstration...")

    # Create random noise images
    fake_images = np.random.rand(num_images, 1, 32, 32)

    # Visualize the fake images
    print("Visualizing fake results...")
    visualize_generated_images(fake_images, save_path="fake_samples.png")

    print(
        "Note: These are not real PixelCNN generations. They are random noise for demonstration."
    )
    print(
        "To generate real images, train the model first with: python ar/pixelcnn_train.py"
    )


if __name__ == "__main__":
    # Try to generate images using a trained model
    if os.path.exists("model.safetensors"):
        generate_random_images(num_images=1, temperature=0.8)
    else:
        # If no model is available, generate fake images for demonstration
        print(
            "No trained model found. Generating random noise images for demonstration."
        )
        print("To train a model first, run: python ar/pixelcnn_train.py")
        generate_and_visualize_fake_images(num_images=10)
