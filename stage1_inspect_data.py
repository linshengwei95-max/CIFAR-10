"""阶段1：认识 CIFAR-10，只查看数据，不训练模型。"""

from pathlib import Path

import matplotlib

# 把图片保存到文件，方便在 Mac 和 Windows 上打开。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


# 路径根据当前脚本的位置确定，兼容两台电脑的不同项目路径。
project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"
output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

# 把图片转换成 [通道, 高度, 宽度] 的浮点 Tensor。
transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root=str(data_dir), train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root=str(data_dir), train=False, download=True, transform=transform
)

print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))
print("Classes:", train_dataset.classes)
print("Number of classes:", len(train_dataset.classes))

# 从 Dataset 取出一条样本：一张图片和它的类别编号。
image, label = train_dataset[0]
print("Single image shape:", image.shape)
print("Single image dtype:", image.dtype)
print("Single image device:", image.device)
print("Pixel range:", image.min().item(), "to", image.max().item())
print("Single label:", label, "=", train_dataset.classes[label])

# 手动取8条样本，观察它们组成一批后的 shape。
# 下一阶段再用 DataLoader 自动组织 batch。
batch_size = 8
image_list = []
label_list = []
for index in range(batch_size):
    sample_image, sample_label = train_dataset[index]
    image_list.append(sample_image)
    label_list.append(sample_label)

images = torch.stack(image_list, dim=0)
labels = torch.tensor(label_list, dtype=torch.long)
print("images.shape:", images.shape)
print("labels.shape:", labels.shape)
print("labels:", labels)

# 绘图模板：最多显示本批的前8张图片。
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for index, ax in enumerate(axes.flat):
    if index < len(images):
        class_id = labels[index].item()
        # imshow 使用 [高度, 宽度, 通道]，这里只调整显示时的维度顺序。
        ax.imshow(images[index].permute(1, 2, 0).numpy(), interpolation="nearest")
        ax.set_title(f"{class_id}: {train_dataset.classes[class_id]}")
    ax.axis("off")

fig.suptitle("CIFAR-10: image and true label")
fig.tight_layout()
image_path = output_dir / "cifar10_preview.png"
fig.savefig(image_path, dpi=160)
plt.close(fig)
print("Preview saved to:", image_path)
