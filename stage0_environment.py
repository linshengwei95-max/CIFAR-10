"""阶段0：检查环境，运行一个简单的 Tensor 示例。"""

import sys

import matplotlib
import torch
import torchvision


# 只检查环境，不加载数据、不建立模型。
print("Python:", sys.version.split()[0])
print("Python executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("matplotlib:", matplotlib.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())

# 按可用性选择设备：NVIDIA GPU → Apple GPU → CPU。
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Selected device:", device)

# 输入：2行3列的浮点数。默认先在 CPU 上创建。
x = torch.tensor(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    dtype=torch.float32,
)
print("x device before .to():", x.device)

# .to() 返回目标设备上的 Tensor，因此把返回值赋给 x。
x = x.to(device)

# 对每个元素乘2再加1，不改变 shape。
y = x * 2 + 1

print("x:", x)
print("x.shape:", x.shape)
print("x.device:", x.device)
print("y:", y)
print("y.shape:", y.shape)
print("y.device:", y.device)
