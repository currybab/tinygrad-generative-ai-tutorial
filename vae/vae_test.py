from vae_train import VAEModel, EncoderModel, DecoderModel
from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad.nn import datasets
from tinygrad import Tensor, dtypes
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    encoder = EncoderModel()
    decoder = DecoderModel()

    model = VAEModel(encoder, decoder)

    # and load it back in
    state_dict = safe_load("model.safetensors")
    load_state_dict(model, state_dict)

    X_train, Y_train, X_test, Y_test = datasets.mnist(fashion=True)
    X_test = X_test.div(255.0).pad(((0, 0), (0, 0), (2, 2), (2, 2)))

    # test the model
    samples = Tensor.randint(10, high=X_test.shape[0])
    result, _, _ = model(X_test[samples])

    # visualize result
    print(result.shape)
    for i in range(int(result.shape[0])):
        # 원본 이미지와 생성된 이미지를 나란히 표시
        plt.figure(figsize=(10, 5))

        # 원본 이미지
        plt.subplot(1, 2, 1)
        original = X_test[samples[i]].reshape(32, 32)
        original = original.mul(255).clip(0, 255).cast(dtypes.uint8)
        plt.imshow(original.numpy(), cmap="gray")
        plt.title("Original")

        # 생성된 이미지
        plt.subplot(1, 2, 2)
        r = result[i].reshape(32, 32)
        r = r.mul(255).clip(0, 255).cast(dtypes.uint8)
        plt.imshow(r.numpy(), cmap="gray")
        plt.title("Generated")

        plt.tight_layout()
        plt.show()
