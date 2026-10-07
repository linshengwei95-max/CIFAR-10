"""阶段10：观察一批训练图片的ResNet18输出与模型参数量；不训练或保存权重。"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.resnet18_cifar import build_resnet18_cifar
from models.simple_cnn import SimpleCNN
from stage9_train_ab import load_saved_split


def main():
    project_dir = Path(__file__).resolve().parent
    plain_data = datasets.CIFAR10(
        root=str(project_dir / "data"), train=True, download=False,
        transform=transforms.ToTensor(),
    )
    train_indices, _, _ = load_saved_split(
        plain_data, project_dir / "outputs" / "stage9_split_seed42.json",
    )
    loader = DataLoader(
        Subset(plain_data, train_indices), batch_size=8,
        shuffle=False, num_workers=0,
    )
    images, labels = next(iter(loader))  # 只取固定训练划分中的一批图片。

    torch.manual_seed(42)
    simple_model = SimpleCNN()  # 只用于统计参数量，不执行旧模型的forward。
    model = build_resnet18_cifar()
    simple_count = sum(p.numel() for p in simple_model.parameters())
    resnet_count = sum(p.numel() for p in model.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    images = images.to(device)
    model.eval()  # 固定BatchNorm评估行为，避免本次观察更新其运行统计。
    with torch.no_grad():
        outputs = model(images)  # [8, 3, 32, 32] -> [8, 10]。

    print("Device:", device)
    print("Training images in saved split:", len(train_indices))
    print("images.shape:", images.shape)
    print("labels.shape:", labels.shape)
    print("outputs.shape:", outputs.shape)
    print("outputs device:", outputs.device)
    print("outputs.requires_grad:", outputs.requires_grad)
    print("All logits finite:", torch.isfinite(outputs).all().item())
    print(f"SimpleCNN parameters: {simple_count:,}")
    print(f"CIFAR ResNet18 parameters: {resnet_count:,}")
    print(f"Parameter ratio (ResNet18 / SimpleCNN): {resnet_count / simple_count:.2f}")
    print("Scope: one untrained-model forward; no training, accuracy, timing or official test evaluation.")


if __name__ == "__main__":
    main()
