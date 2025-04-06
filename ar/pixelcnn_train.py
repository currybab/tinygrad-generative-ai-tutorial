import math
from tinygrad import nn, dtypes, TinyJit, GlobalCounters
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_state_dict, safe_save
from tinygrad.helpers import trange

# PixelCNN
#
# pixel order
# 1. from left to right
# 2. from top to bottom
PIXEL_LEVELS = 4


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in {"A", "B"}
        self.mask_type = mask_type

    def __call__(self, x: Tensor) -> Tensor:
        mask = Tensor.ones(self.weight.shape, requires_grad=False).contiguous()
        _, _, kH, kW = self.weight.shape
        mask[:, :, kH // 2, kW // 2 + (self.mask_type == "B") :] = 0.0
        mask[:, :, kH // 2 + 1 :] = 0.0
        return x.conv2d(
            self.weight * mask,
            self.bias,
            self.groups,
            self.stride,
            self.dilation,
            self.padding,
        )


class ResidualBlock:
    def __init__(self, in_channels, out_channels):
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
        )
        self.masked_conv = MaskedConv2d(
            mask_type="B",
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )

    def __call__(self, x: Tensor) -> Tensor:
        return self.conv2(self.masked_conv(self.conv1(x).relu()).relu()).relu()


class PixelCNN:
    def __init__(self):
        self.masked_conv = MaskedConv2d(
            mask_type="A",
            in_channels=1,
            out_channels=128,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        self.residuals = [ResidualBlock(128, 128) for _ in range(5)]
        self.masked_conv2 = [
            MaskedConv2d(
                mask_type="B",
                in_channels=128,
                out_channels=128,
                kernel_size=1,
                stride=1,
            )
            for _ in range(2)
        ]
        self.conv = nn.Conv2d(
            in_channels=128, out_channels=PIXEL_LEVELS, kernel_size=1, stride=1
        )

    def __call__(self, x: Tensor) -> Tensor:
        y = self.masked_conv(x).relu()
        for residual in self.residuals:
            y = residual(y)
        for masked_conv in self.masked_conv2:
            y = masked_conv(y).relu()
        return self.conv(y)


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = nn.datasets.mnist(fashion=True)

    # preprocess: normalize and pad to 32x32
    def preprocess(x: Tensor) -> tuple[Tensor, Tensor]:
        x_cut: Tensor = x.pad(((0, 0), (0, 0), (2, 2), (2, 2))).avg_pool2d((2, 2), 2)
        x_int = x_cut.div(256 / PIXEL_LEVELS).cast(dtypes.int)
        x_float = x_int.div(PIXEL_LEVELS)
        return x_float, x_int

    X_train, X_target = preprocess(X_train)
    # X_test = (
    #     X_test.div(256 / PIXEL_LEVELS)
    #     .cast(dtypes.float32)
    #     .div(PIXEL_LEVELS)
    #     .pad(((0, 0), (0, 0), (2, 2), (2, 2)))
    # )

    train_size = X_train.shape[0]
    batch_size = 100
    step_size = math.ceil(train_size / batch_size)
    epochs = 150
    print(
        f"train_size={train_size}, batch_size={batch_size}, step_size={step_size}, epochs={epochs}"
    )

    model = PixelCNN()
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=0.0005)

    @TinyJit
    @Tensor.train()
    def train_step(samples: Tensor) -> tuple[Tensor, Tensor]:
        output = model(X_train[samples]).permute(0, 2, 3, 1)
        loss = output.sparse_categorical_crossentropy(X_target[samples].squeeze(1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.realize(), output

    for step in (t := trange(int(epochs * step_size))):
        GlobalCounters.reset()
        batch = batch_size * (i := step % step_size)
        samples = Tensor.arange(batch, batch + batch_size)
        train_loss, output = train_step(samples)
        t.set_description(
            f"loss: {train_loss.item():6.4f}, 8,8={output[0, 8, 8].numpy()}, real={X_target[0, 0, 8, 8].numpy()}"
        )

    # first we need the state dict of our model
    state_dict = get_state_dict(model)

    # then we can just save it to a file
    safe_save(state_dict, "model.safetensors")
