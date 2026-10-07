"""阶段5：从随机初始化开始训练1个epoch，观察训练loss，不保存权重。"""

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
    batch_size = 8
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 同一次训练的所有batch共用这个模型和优化器。
    model = SimpleCNN().to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Device:", device)
    print("Training images:", len(train_dataset))
    print("batch_size:", batch_size)
    print("Batches in this epoch:", len(train_loader))

    running_loss = 0.0
    seen_images = 0
    # 逐批取图片和对应标签，一次完整遍历就是1个epoch。
    for step, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)  # [B, 3, 32, 32] -> [B, 10]
        loss = criterion(outputs, labels)  # [B, 10] + [B] -> 标量loss
        loss.backward()
        optimizer.step()

        # loss默认是本批均值；乘本批样本数后才能按样本统计整轮均值。
        batch_loss = loss.item()
        running_loss += batch_loss * images.size(0)
        seen_images += images.size(0)

        if step == 1 or step % 500 == 0 or step == len(train_loader):
            mean_loss = running_loss / seen_images
            print(
                f"Batch {step}/{len(train_loader)} | "
                f"batch loss: {batch_loss:.4f} | "
                f"mean training loss so far: {mean_loss:.4f}",
                flush=True,
            )

    print("Epochs completed: 1")
    print("Training images seen:", seen_images)
    print("Optimizer steps:", len(train_loader))
    print("Mean training loss:", running_loss / seen_images)


if __name__ == "__main__":
    main()
