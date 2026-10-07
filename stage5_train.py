"""阶段5：从随机初始化连续训练5个epoch，结束时保存最后的模型参数。"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.simple_cnn import SimpleCNN


def main():
    project_dir = Path(__file__).resolve().parent
    train_dataset = datasets.CIFAR10(
        root=str(project_dir / "data"), train=True, download=False,
        transform=transforms.ToTensor(),
    )
    batch_size = 8
    num_epochs = 5
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 所有epoch共用同一个模型和优化器，不在每轮重新随机初始化。
    model = SimpleCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Device:", device)
    print("Training images:", len(train_dataset))
    print("batch_size:", batch_size)
    print("Batches per epoch:", len(train_loader))
    print("Number of epochs:", num_epochs)

    for epoch in range(1, num_epochs + 1):
        model.train()
        # 每轮分别统计平均loss，参数继续沿用上一轮的学习结果。
        running_loss = 0.0
        seen_images = 0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)  # [B, 3, 32, 32] -> [B, 10]
            loss = criterion(outputs, labels)  # [B, 10] + [B] -> 标量loss
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss * images.size(0)
            seen_images += images.size(0)

            if step == 1 or step % 1000 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Batch {step}/{len(train_loader)} | "
                    f"batch loss: {batch_loss:.4f} | "
                    f"mean training loss so far: {running_loss / seen_images:.4f}",
                    flush=True,
                )

        print(
            f"Epoch {epoch}/{num_epochs} completed | "
            f"Mean training loss: {running_loss / seen_images:.6f}",
            flush=True,
        )

    # 保存最后一轮的参数，下一阶段可加载来评估；当前不做评估。
    checkpoint_dir = project_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "stage5_last.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print("Epochs completed:", num_epochs)
    print("Optimizer steps:", num_epochs * len(train_loader))
    print("Final model parameters saved to:", checkpoint_path)


if __name__ == "__main__":
    main()
