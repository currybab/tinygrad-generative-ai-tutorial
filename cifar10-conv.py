from tinygrad.nn import datasets
from tinygrad import Tensor, nn, TinyJit, GlobalCounters
from tinygrad.helpers import getenv, colored, trange


X_train, Y_train, X_test, Y_test = datasets.cifar()
X_train = X_train.div(255.0)
X_test = X_test.div(255.0)

train_size = X_train.shape[0]
batch_size = getenv("BS", 32)
step_size = getenv("STEPS", 10 * train_size // batch_size)


class Model:
    def __init__(self):
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1
        )
        self.norm1 = nn.BatchNorm(32)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.BatchNorm(64)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1
        )
        self.norm3 = nn.BatchNorm(64)
        self.dense1 = nn.Linear(64 * 8 * 8, 128)
        self.norm4 = nn.BatchNorm(128)
        self.dense2 = nn.Linear(128, 10)

    def __call__(self, x: Tensor) -> Tensor:
        y1 = self.norm1(self.conv1(x)).leaky_relu()
        y2 = self.norm2(self.conv2(y1)).leaky_relu()
        y3 = self.norm3(self.conv3(y2)).leaky_relu()
        y4 = self.norm4(self.dense1(y3.flatten(start_dim=1))).leaky_relu().dropout()
        y5 = self.dense2(y4).softmax(1)
        return y5


if __name__ == "__main__":
    model = Model()
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=0.0005)

    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        opt.zero_grad()
        samples = Tensor.randint(batch_size, high=train_size)
        loss = (
            model(X_train[samples])
            .sparse_categorical_crossentropy(Y_train[samples])
            .backward()
        )
        opt.step()
        return loss.realize()

    @TinyJit
    @Tensor.test()
    def get_test_acc() -> Tensor:
        return (model(X_test).argmax(axis=1) == Y_test).mean().realize() * 100

    test_acc = float("nan")
    for i in (t := trange(step_size)):
        GlobalCounters.reset()  # NOTE: this makes it nice for DEBUG=2 timing
        loss = train_step()
        if i % (train_size // batch_size) == 0:
            test_acc = get_test_acc().item()
        t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

    # verify eval acc
    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(colored(f"{test_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"{test_acc=} < {target}", "red"))
