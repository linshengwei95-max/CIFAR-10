"""阶段5第一小步：一批图片做 forward 并计算 loss，不反向传播、不更新参数。"""

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
    images = images.to(device)
    labels = labels.to(device)

    # 输入为每张图片的10类原始分数，以及对应的真实类别编号。
    criterion = torch.nn.CrossEntropyLoss()
    # 为后续 backward 保留计算关系，forward 不包在 no_grad 中。
    outputs = model(images)  # [B, 3, 32, 32] -> [B, 10]
    loss = criterion(outputs, labels)  # [B, 10] + [B] -> 标量，默认取均值

    print("Device:", device)
    print("images:", images.shape, images.dtype, images.device)
    print("labels:", labels.shape, labels.dtype, labels.device)
    print("outputs:", outputs.shape, outputs.device)
    print("outputs.requires_grad:", outputs.requires_grad)
    print("loss.shape:", loss.shape)
    print("loss.requires_grad:", loss.requires_grad)
    print("loss:", loss.item())
    print("All parameter gradients are None:", all(p.grad is None for p in model.parameters()))
    print("Only forward and loss; no backward, optimizer or parameter updates.")


if __name__ == "__main__":
    main()
