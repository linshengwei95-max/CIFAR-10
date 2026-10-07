"""阶段4：一批真实图片做前向计算，观察 logits；不训练、不保存权重。"""

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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    images, labels = next(iter(train_loader))

    # 模型和图片都留在 CPU 上，当前阶段无需切换设备。
    model = SimpleCNN()
    # 本次只观察输出，关闭梯度记录；没有参数更新操作。
    with torch.no_grad():
        outputs = model(images)  # [B, 3, 32, 32] -> [B, 10]

    # 在每张图片的10个分数中，取最大值所在的类别位置。
    predicted_labels = outputs.argmax(dim=1)  # [B, 10] -> [B]

    print("images.shape:", images.shape)
    print("labels.shape:", labels.shape)
    print("outputs.shape:", outputs.shape)
    print("outputs device:", outputs.device)
    print("First image logits:", outputs[0])
    print("Class order:", train_dataset.classes)
    print("predicted_labels.shape:", predicted_labels.shape)
    print("True labels:", labels)
    print("Predicted labels:", predicted_labels)
    print("First true class:", train_dataset.classes[labels[0].item()])
    print("First predicted class:", train_dataset.classes[predicted_labels[0].item()])
    print("Model is untrained; this run does not measure classification quality.")


if __name__ == "__main__":
    main()
