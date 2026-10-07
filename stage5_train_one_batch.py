"""阶段5：训练一批8张图片，只执行一次参数更新，不保存权重。"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.simple_cnn import SimpleCNN


def main():
    data_dir = Path(__file__).resolve().parent / "data"
    train_dataset = datasets.CIFAR10(
        root=str(data_dir), train=True, download=False,
        transform=transforms.ToTensor(),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=0,
    )
    images, labels = next(iter(train_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.train()  # 设置训练模式，这句本身不会更新参数。
    images = images.to(device)
    labels = labels.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    # 优化器管理该模型的参数；lr 是控制更新幅度的学习率。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 保存最后分类层的bias作比较，只观察这10个参数是否变化。
    tracked_parameter = model.classifier[-1].bias
    parameter_before = tracked_parameter.detach().clone()

    optimizer.zero_grad(set_to_none=True)  # 清除旧梯度，避免逐批更新时累积。
    outputs = model(images)  # [8, 3, 32, 32] -> [8, 10]，保留计算关系。
    loss = criterion(outputs, labels)  # [8, 10] + [8] -> 标量loss。
    loss.backward()  # 计算参数梯度，此时参数还没有被更新。
    gradient_norm = tracked_parameter.grad.norm().item()
    optimizer.step()  # 使用梯度，执行这一次参数更新。

    max_parameter_change = (
        tracked_parameter.detach() - parameter_before
    ).abs().max().item()

    print("Device:", device)
    print("images:", images.shape, images.device)
    print("labels:", labels.shape, labels.device)
    print("outputs:", outputs.shape, outputs.device)
    print("Loss before update:", loss.item())
    print("Tracked parameter: final classifier bias, shape", tracked_parameter.shape)
    print("Gradient norm:", gradient_norm)
    print("Max parameter change:", max_parameter_change)
    print("Tracked parameter changed:", max_parameter_change > 0)
    print("Optimizer steps: 1; this script trains exactly one batch.")


if __name__ == "__main__":
    main()
