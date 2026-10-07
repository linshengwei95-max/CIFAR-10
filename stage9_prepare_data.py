"""阶段9：固定训练/验证划分，检查数据并预览增强；不创建或训练模型。"""

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def make_split(labels, seed=42):
    """输入 [50000] 类别编号，返回两个互不重叠的原始图片索引列表。"""
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = [], []
    for class_id in range(10):
        class_indices = (labels == class_id).nonzero(as_tuple=True)[0]
        order = torch.randperm(len(class_indices), generator=generator)
        class_indices = class_indices[order]
        val_indices.extend(class_indices[:1000].tolist())
        train_indices.extend(class_indices[1000:].tolist())

    # 混合不同类别；后续训练仍可以在固定训练索引内逐轮shuffle。
    train_indices = torch.tensor(train_indices)[
        torch.randperm(len(train_indices), generator=generator)
    ].tolist()
    val_indices = torch.tensor(val_indices)[
        torch.randperm(len(val_indices), generator=generator)
    ].tolist()
    return train_indices, val_indices


def check_split(labels, train_indices, val_indices):
    """在保存或复用划分前，检查索引、覆盖范围和各类数量。"""
    if len(labels) != 50000 or torch.bincount(labels, minlength=10).tolist() != [5000] * 10:
        raise ValueError("数据问题：预期官方训练集每类5000张，共50000张。")
    for indices, expected_size in ((train_indices, 40000), (val_indices, 10000)):
        if len(indices) != expected_size:
            raise ValueError(f"划分问题：预期{expected_size}个索引，实际{len(indices)}个。")
        if any(type(index) is not int or not 0 <= index < len(labels) for index in indices):
            raise ValueError("划分问题：索引必须是范围内的整数。")
        if len(set(indices)) != len(indices):
            raise ValueError("划分问题：同一集合中有重复索引。")
    train_set, val_set = set(train_indices), set(val_indices)
    if train_set & val_set or train_set | val_set != set(range(len(labels))):
        raise ValueError("划分问题：训练/验证有重叠，或未完整覆盖原始图片。")
    train_counts = torch.bincount(labels[train_indices], minlength=10)
    val_counts = torch.bincount(labels[val_indices], minlength=10)
    if train_counts.tolist() != [4000] * 10 or val_counts.tolist() != [1000] * 10:
        raise ValueError("划分问题：预期每类4000张训练、1000张验证。")
    return train_counts, val_counts


def main():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    split_path = output_dir / "stage9_split_seed42.json"
    plain_transform = transforms.ToTensor()
    augmented_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    # 分别配置transform，避免训练和验证共用一个可变的增强设置。
    plain_dataset = datasets.CIFAR10(
        root=str(project_dir / "data"), train=True, download=False,
        transform=plain_transform,
    )
    augmented_dataset = datasets.CIFAR10(
        root=str(project_dir / "data"), train=True, download=False,
        transform=augmented_transform,
    )
    labels = torch.tensor(plain_dataset.targets, dtype=torch.long)  # [50000]
    metadata = {
        "schema_version": 1,
        "dataset": "CIFAR10",
        "official_split": "train",
        "dataset_size": len(plain_dataset),
        "classes": plain_dataset.classes,
        "targets_sha256": hashlib.sha256(bytes(plain_dataset.targets)).hexdigest(),
        "split_seed": 42,
        "validation_per_class": 1000,
    }
    if split_path.exists():
        # 已有文件直接复用；不在每次运行时悄悄换一批验证图片。
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if any(split.get(key) != value for key, value in metadata.items()):
            raise ValueError("划分文件与当前数据或配置不符，请先核对该文件。")
        train_indices, val_indices = split["train_indices"], split["val_indices"]
        split_status = "reused"
    else:
        train_indices, val_indices = make_split(labels, seed=metadata["split_seed"])
        split = {**metadata, "train_indices": train_indices, "val_indices": val_indices}
        split_status = "created"

    train_counts, val_counts = check_split(labels, train_indices, val_indices)
    if split_status == "created":
        split_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

    train_dataset = Subset(augmented_dataset, train_indices)
    val_dataset = Subset(plain_dataset, val_indices)
    report_lines = [
        f"Python: {sys.executable}",
        f"Split file: {split_path} ({split_status})",
        "Split seed: 42",
        f"Training images: {len(train_dataset)}",
        f"Validation images: {len(val_dataset)}",
        "Index overlap: 0 | Full coverage: 50000/50000 | Duplicate indices: 0",
        "Class           Train  Validation",
    ]
    for class_id, name in enumerate(plain_dataset.classes):
        report_lines.append(f"{name:14s}  {train_counts[class_id].item():5d}  {val_counts[class_id].item():10d}")

    # 这里仅固定取一批检查shape；正式训练的shuffle留到训练脚本。
    torch.manual_seed(123)  # 预览用随机数，不参与已经固定的划分。
    for name, dataset, indices in (
        ("Train (augmented)", train_dataset, train_indices),
        ("Validation (plain)", val_dataset, val_indices),
    ):
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        images, batch_labels = next(iter(loader))
        if images.shape != (8, 3, 32, 32) or batch_labels.shape != (8,):
            raise ValueError(f"Tensor维度问题：{name}的batch shape不符合预期。")
        if images.dtype != torch.float32 or batch_labels.dtype != torch.int64:
            raise ValueError(f"Tensor类型问题：{name}的图片或标签类型不符合预期。")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError(f"数据问题：{name}的图片包含无效数值。")
        if not torch.equal(batch_labels, labels[indices[:8]]):
            raise ValueError(f"数据问题：{name}的标签与原图索引不匹配。")
        report_lines.append(
            f"{name}: images={list(images.shape)}, {images.dtype}, {images.device}; "
            f"labels={list(batch_labels.shape)}, {batch_labels.dtype}; "
            f"range=[{images.min().item():.3f}, {images.max().item():.3f}]"
        )
    if not torch.equal(val_dataset[0][0], val_dataset[0][0]):
        raise ValueError("验证读取问题：同一张验证图片重复读取发生变化。")
    report_lines.append("Repeated validation read: identical")

    # 从训练划分里选汽车和船；每行原图加三个随机增强版本。
    preview_indices = [
        next(index for index in train_indices if plain_dataset.targets[index] == class_id)
        for class_id in (1, 8)
    ]
    torch.manual_seed(123)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for row, index in enumerate(preview_indices):
        original, label = plain_dataset[index]
        views = [(original, label)] + [augmented_dataset[index] for _ in range(3)]
        for column, (image, view_label) in enumerate(views):
            if view_label != label or image.shape != (3, 32, 32):
                raise ValueError("增强问题：图片尺寸或类别标签改变。")
            ax = axes[row, column]
            ax.imshow(image.permute(1, 2, 0).numpy(), interpolation="nearest")
            title = "Original" if column == 0 else f"Crop + Flip {column}"
            ax.set_title(f"{title}\n{plain_dataset.classes[label]} (label={label})")
            ax.axis("off")
    fig.suptitle("CIFAR-10: same training image, random augmented views")
    fig.tight_layout()
    preview_path = output_dir / "stage9_augmentation_preview.png"
    fig.savefig(preview_path, dpi=160)
    plt.close(fig)
    report_lines.extend([
        f"Preview training indices: {preview_indices}",
        f"Preview saved to: {preview_path}",
        "Scope: data preparation and preview only; no model, training, or test evaluation.",
    ])
    report = "\n".join(report_lines) + "\n"
    (output_dir / "stage9_data_check.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
