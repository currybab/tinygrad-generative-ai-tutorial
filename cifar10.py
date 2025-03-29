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
        self.layer1 = nn.Linear(3072, 200)
        self.layer2 = nn.Linear(200, 150)
        self.layer3 = nn.Linear(150, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x1 = x.flatten(start_dim=1)
        y1 = self.layer1(x1).relu()
        y2 = self.layer2(y1).relu()
        y3 = self.layer3(y2).softmax(1)
        return y3


if __name__ == "__main__":
    model = Model()
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr = 0.0005)
    
    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        opt.zero_grad()
        samples = Tensor.randint(batch_size, high=train_size)
        loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
        opt.step()
        return loss.realize()
    

    @TinyJit
    @Tensor.test()
    def get_test_acc() -> Tensor:
        return (model(X_test).argmax(axis=1) == Y_test).mean().realize()*100

    test_acc = float('nan')
    for i in (t:=trange(step_size)):
        GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
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

