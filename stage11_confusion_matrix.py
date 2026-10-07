"""阶段11：统计完整验证集的类别混淆；--analyze才执行已有模型推理。"""

import argparse
from datetime import datetime
import json
from pathlib import Path


SOURCE_RUN = "stage10_compare_20261007_181857_108622"
BATCH_SIZE = 8
TOP_ERRORS_TO_SHOW = 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", action="store_true", help="加载已有best权重，统计全部10000张验证图片")
    args = parser.parse_args()
    print(f"Model: CIFAR ResNet18 best | Source run: {SOURCE_RUN}")
    print(f"Scope: full saved validation split (10000 images); batch_size={BATCH_SIZE}")
    print("Matrix: rows = true classes; columns = predicted classes")
    if not args.analyze:
        print("Analysis not started. Add --analyze to run the validation analysis.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from models.resnet18_cifar import build_resnet18_cifar
    from stage9_train_ab import load_saved_split, write_json

    project_dir = Path(__file__).resolve().parent
    source_dir = project_dir / "outputs" / SOURCE_RUN
    checkpoint_path = project_dir / "checkpoints" / SOURCE_RUN / "CIFAR_ResNet18_best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"数据/路径问题：未找到已有best权重：{checkpoint_path}")
    config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((source_dir / "CIFAR_ResNet18_metrics.json").read_text(encoding="utf-8"))
    plain_data = datasets.CIFAR10(
        root=str(project_dir / "data"), train=True, download=False,
        transform=transforms.ToTensor(),
    )
    _, val_indices, split_hash = load_saved_split(
        plain_data, project_dir / "outputs" / "stage9_split_seed42.json",
    )
    if split_hash != config["split_sha256"]:
        raise ValueError("数据/划分问题：当前索引与所选训练实验不一致。")
    loader = DataLoader(
        Subset(plain_data, val_indices), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(46),
    )
    classes = plain_data.classes
    num_classes = len(classes)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)  # [10, 10]计数表。

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_resnet18_cifar()
    saved_parameters = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved_parameters, strict=True)
    model = model.to(device)
    model.eval()
    print("Device:", device)
    print("Loaded best checkpoint:", checkpoint_path)
    print("Selected best epoch:", metrics["best_epoch"])
    images_seen = 0
    with torch.no_grad():
        for step, (images, labels) in enumerate(loader, start=1):
            logits = model(images.to(device))  # [B, 3, 32, 32] -> [B, 10]。
            if not torch.isfinite(logits).all().item():
                raise ValueError("模型输出问题：发现非有限logits，停止分析。")
            predicted = logits.argmax(dim=1).cpu()  # [B]，与CPU标签逐张对应。
            for true_id, predicted_id in zip(labels.tolist(), predicted.tolist()):
                confusion[true_id, predicted_id] += 1  # 真实类别这一行、预测类别这一列加1。
            images_seen += len(labels)
            if step == 1:
                print("First batch shapes:", images.shape, labels.shape, logits.shape, predicted.shape)
            if step == 1 or step % 250 == 0 or step == len(loader):
                print(f"Validation analysis | Batch {step}/{len(loader)} | Images counted: {images_seen}", flush=True)

    total = confusion.sum().item()
    correct = confusion.diag().sum().item()  # 对角线的计数相加，就是分对的总数。
    if total != len(val_indices):
        raise ValueError("数据/计数问题：混淆矩阵总数与验证集长度不一致。")
    per_class = []
    for class_id, name in enumerate(classes):
        row_total = confusion[class_id].sum().item()
        class_correct = confusion[class_id, class_id].item()
        per_class.append({
            "class_id": class_id, "class": name, "images": row_total,
            "correct": class_correct, "recall": class_correct / row_total,
        })
    # 只给非对角线的错误方向排序；两个相反方向分别计数。
    error_pairs = [
        {"true_class": classes[i], "predicted_class": classes[j], "count": confusion[i, j].item()}
        for i in range(num_classes) for j in range(num_classes)
        if i != j and confusion[i, j].item() > 0
    ]
    error_pairs.sort(key=lambda row: (-row["count"], row["true_class"], row["predicted_class"]))

    output_dir = project_dir / "outputs" / f"stage11_confusion_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "confusion.json", {
        "source_run": SOURCE_RUN, "model": "CIFAR_ResNet18", "best_epoch": metrics["best_epoch"],
        "checkpoint": str(checkpoint_path.relative_to(project_dir)), "split_sha256": split_hash,
        "device": str(device), "batch_size": BATCH_SIZE, "transform": "ToTensor", "shuffle": False,
        "classes": classes, "matrix_rows": "true class", "matrix_columns": "predicted class",
        "confusion_matrix": confusion.tolist(), "images_counted": total, "correct": correct,
        "errors": total - correct, "validation_accuracy": correct / total,
        "source_best_validation_accuracy": metrics["best_val_accuracy"],
        "per_class": per_class, "error_pairs_in_count_order": error_pairs,
        "scope": "Full fixed validation split; existing best weights; no parameter updates or official test-set use",
    })

    # 绘图模板：每个格子显示实际图片数量；颜色越深，数量越多。
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = ax.imshow(confusion.numpy(), cmap="Blues", vmin=0)
    ax.set_xticks(range(num_classes), classes, rotation=45, ha="right")
    ax.set_yticks(range(num_classes), classes)
    ax.set_xlabel("Predicted class (columns)")
    ax.set_ylabel("True class (rows)")
    ax.set_title(f"CIFAR ResNet18: validation confusion matrix\n{total} images, accuracy {correct / total:.2%}")
    for i in range(num_classes):
        for j in range(num_classes):
            count = confusion[i, j].item()
            ax.text(j, i, str(count), ha="center", va="center", fontsize=9,
                    color="white" if count > confusion.max().item() / 2 else "black")
    fig.colorbar(heatmap, ax=ax, label="Number of images")
    fig.tight_layout()
    figure_path = output_dir / "confusion_matrix.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print("Confusion matrix shape:", confusion.shape)
    print(f"Validation images counted: {total} | Correct: {correct} | Errors: {total - correct}")
    print(f"Validation accuracy: {correct / total:.2%}")
    print(f"Source recorded best validation accuracy: {metrics['best_val_accuracy']:.2%}")
    print("Per-class recall (correct / true-class images):")
    for row in per_class:
        print(f"{row['class']}: {row['correct']}/{row['images']} = {row['recall']:.2%}")
    print(f"Top {TOP_ERRORS_TO_SHOW} directed error pairs:")
    for row in error_pairs[:TOP_ERRORS_TO_SHOW]:
        print(f"{row['true_class']} -> {row['predicted_class']}: {row['count']}")
    print("Confusion figure saved to:", figure_path)
    print("Confusion records saved to:", output_dir / "confusion.json")


if __name__ == "__main__":
    main()
