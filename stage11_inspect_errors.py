"""阶段11：观察固定验证样本中的错误图片；--inspect才执行模型推理。"""

import argparse
from datetime import datetime
import json
from pathlib import Path


SOURCE_RUN = "stage10_compare_20261007_181857_108622"
BATCH_SIZE = 8
NUM_IMAGES = 64
MAX_ERRORS_TO_SHOW = 6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="加载已有best权重，观察验证集前64张图片")
    args = parser.parse_args()
    print(f"Model: CIFAR ResNet18 best | Source run: {SOURCE_RUN}")
    print(f"Scope: first {NUM_IMAGES} saved validation images; batch_size={BATCH_SIZE}")
    print(f"Display: up to {MAX_ERRORS_TO_SHOW} errors in validation order")
    if not args.inspect:
        print("Inspection not started. Add --inspect to run the observation.")
        return

    # 运行开关之后才导入框架；默认模式不读取图片、创建模型或写文件。
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
    selected_indices = val_indices[:NUM_IMAGES]
    loader = DataLoader(
        Subset(plain_data, selected_indices), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(47),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_cifar()
    saved_parameters = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved_parameters, strict=True)
    model = model.to(device)
    model.eval()
    print("Device:", device)
    print("Loaded best checkpoint:", checkpoint_path)
    print("Selected best epoch:", metrics["best_epoch"])

    records, error_examples = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))  # [B, 3, 32, 32] -> [B, 10]。
            if not torch.isfinite(logits).all().item():
                raise ValueError("模型输出问题：发现非有限logits，停止观察。")
            predicted = logits.argmax(dim=1).cpu()  # 每张图片一个预测编号，[B]。
            wrong = predicted != labels  # [B]的布尔Tensor，True表示这张分错。
            offset = len(records)
            if offset == 0:
                print("First batch shapes:", images.shape, labels.shape, logits.shape, wrong.shape)
            for i in range(len(labels)):
                true_id, predicted_id = labels[i].item(), predicted[i].item()
                row = {
                    "validation_position": offset + i,
                    "source_index": selected_indices[offset + i],
                    "true_id": true_id, "true_class": plain_data.classes[true_id],
                    "predicted_id": predicted_id, "predicted_class": plain_data.classes[predicted_id],
                    "is_error": wrong[i].item(),
                }
                records.append(row)
                if row["is_error"] and len(error_examples) < MAX_ERRORS_TO_SHOW:
                    error_examples.append((images[i].clone(), row))

    error_count = sum(row["is_error"] for row in records)
    output_dir = project_dir / "outputs" / f"stage11_errors_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    preview_path = None
    if error_examples:
        rows = (len(error_examples) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(10, 2.5 * rows))
        for i, ax in enumerate(axes.flat):
            if i < len(error_examples):
                image, row = error_examples[i]
                ax.imshow(image.permute(1, 2, 0).numpy(), interpolation="nearest")
                ax.set_title(f"True: {row['true_class']}\nPred: {row['predicted_class']}\nIndex: {row['source_index']}", fontsize=10)
            ax.axis("off")
        fig.suptitle(f"CIFAR ResNet18: first {len(error_examples)} errors among {len(records)} validation images")
        fig.tight_layout()
        preview_path = output_dir / "error_examples.png"
        fig.savefig(preview_path, dpi=160)
        plt.close(fig)
    else:
        print("No errors in these observed images; no error preview generated.")

    write_json(output_dir / "inspection.json", {
        "source_run": SOURCE_RUN, "model": "CIFAR_ResNet18", "best_epoch": metrics["best_epoch"],
        "checkpoint": str(checkpoint_path.relative_to(project_dir)), "split_sha256": split_hash,
        "device": str(device), "transform": "ToTensor", "shuffle": False,
        "images_inspected": len(records), "errors_in_observed_images": error_count,
        "errors_displayed": len(error_examples), "records": records,
        "scope": "First saved validation images only; this sample is for visual observation, not a full-set performance estimate",
    })
    print("Images inspected:", len(records), "| Errors in these images:", error_count)
    for _, row in error_examples:
        print(f"Index {row['source_index']} | True: {row['true_class']} | Predicted: {row['predicted_class']}")
    if preview_path is not None:
        print("Error preview saved to:", preview_path)
    print("Inspection records saved to:", output_dir / "inspection.json")


if __name__ == "__main__":
    main()
