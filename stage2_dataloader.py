"""阶段2：用 DataLoader 组织 batch，只观察数据，不训练模型。"""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main():
    data_dir = Path(__file__).resolve().parent / "data"
    transform = transforms.ToTensor()

    # 复用阶段1已经下载的数据。
    train_dataset = datasets.CIFAR10(
        root=str(data_dir), train=True, download=False, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_dir), train=False, download=False, transform=transform
    )

    batch_size = 8
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print("batch_size:", batch_size)
    print("Training images:", len(train_dataset))
    print("Test images:", len(test_dataset))
    print("Training batches:", len(train_loader))
    print("Test batches:", len(test_loader))

    # iter() 建立逐批读取的入口，next() 从中取出第一批。
    images, labels = next(iter(train_loader))
    print("Train images.shape:", images.shape)
    print("Train labels.shape:", labels.shape)
    print("Train images dtype:", images.dtype)
    print("Train labels dtype:", labels.dtype)
    print("Train images device:", images.device)
    print("Train labels:", labels)

    test_images, test_labels = next(iter(test_loader))
    print("Test images.shape:", test_images.shape)
    print("Test labels.shape:", test_labels.shape)
    print("Test labels:", test_labels)


if __name__ == "__main__":
    main()
