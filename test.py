from tinygrad import Tensor, dtypes, nn
import numpy as np

np.set_printoptions(precision=4)
norm = nn.BatchNorm(3)
t = Tensor.rand(2, 3, 4, 4)
print(t.mean().item(), t.std().item())
t = norm(t)
print(t.mean().item(), t.std().item())
