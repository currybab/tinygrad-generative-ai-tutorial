import math
from tinygrad.nn import datasets
from tinygrad import Tensor, nn, TinyJit, GlobalCounters
from tinygrad.helpers import trange
from tinygrad.nn.state import safe_save, get_state_dict

X_train, Y_train, X_test, Y_test = datasets.mnist(fashion=True)

# preprocess: normalize and pad to 32x32
X_train = X_train.div(255.0).pad(((0, 0), (0, 0), (2, 2), (2, 2)))
X_test = X_test.div(255.0).pad(((0, 0), (0, 0), (2, 2), (2, 2)))

train_size = X_train.shape[0]
batch_size = 100
step_size = math.ceil(train_size / batch_size)
epochs = 5
print(
    f"train_size={train_size}, batch_size={batch_size}, step_size={step_size}, epochs={epochs}"
)


class EncoderModel:
    # Conv2D: output_size = ((input_size + 2 * padding - kernel_size) / stride) + 1
    def __init__(self):
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1
        )  # (32 + 2 * 1 - 3) // 2 + 1 = 16
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
        )  # (16 + 2 * 1 - 3) // 2 + 1 = 8
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        )  # (8 + 2 * 1 - 3) // 2 + 1 = 4
        self.dense = nn.Linear(128 * 4 * 4, 2)

    def __call__(self, x: Tensor) -> Tensor:
        y1 = self.conv1(x).relu()  # n * 32 * 16 * 16
        y2 = self.conv2(y1).relu()  # n * 64 * 8 * 8
        y3 = self.conv3(y2).relu()  # n * 128 * 4 * 4
        return self.dense(y3.flatten(start_dim=1))  # n * 2


class DecoderModel:
    # ConvTranspose2d:  output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
    def __init__(self):
        self.dense = nn.Linear(2, 128 * 4 * 4)
        self.convt1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )  # (4 - 1) * 2 - 2 * 1 + 3 + 1 = 8
        self.convt2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )  # (8 - 1) * 2 - 2 * 1 + 3 + 1 = 16
        self.convt3 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )  # (16 - 1) * 2 - 2 * 1 + 3 + 1 = 32
        self.conv = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=3, padding=1
        )  # (32 + 2 * 1 - 3) // 1 + 1 = 32

    def __call__(self, x: Tensor) -> Tensor:
        y1 = self.dense(x).relu().reshape((-1, 128, 4, 4))  # n * 128 * 4 * 4
        y2 = self.convt1(y1).relu()  # n * 64 * 8 * 8
        y3 = self.convt2(y2).relu()  # n * 32 * 16 * 16
        y4 = self.convt3(y3).relu()  # n * 32 * 32 * 32
        return self.conv(y4).sigmoid()  # n * 1 * 32 * 32


class AutoEncoder:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)


if __name__ == "__main__":
    encoder = EncoderModel()
    decoder = DecoderModel()

    model = AutoEncoder(encoder, decoder)
    opt = nn.optim.Adam(nn.state.get_parameters(model))

    @TinyJit
    @Tensor.train()
    def train_step(samples: Tensor) -> Tensor:
        opt.zero_grad()
        output, _ = model(X_train[samples])
        loss = output.clip(1e-7, 1 - 1e-7).binary_crossentropy(
            X_train[samples]
        )  # add clip for numerical stability
        loss.backward()
        opt.step()
        return loss.realize()

    for step in (t := trange(epochs * step_size)):
        GlobalCounters.reset()
        batch = batch_size * (i := step % step_size)
        samples = Tensor.arange(batch, batch + batch_size)
        train_loss = train_step(samples)
        t.set_description(f"loss: {train_loss.item():6.4f}")

    # first we need the state dict of our model
    state_dict = get_state_dict(model)

    # then we can just save it to a file
    safe_save(state_dict, "model.safetensors")
