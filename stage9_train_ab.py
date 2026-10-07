"""阶段9：复用固定索引，对照无增强/裁剪加翻转；只有 --train 才训练。"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.simple_cnn import SimpleCNN
from stage6_evaluate import evaluate
from stage9_prepare_data import check_split


BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
INIT_SEED = 42
ORDER_SEED = 43
AUGMENT_SEED = 44


def load_saved_split(plain_dataset, split_path):
    """只读取已有索引；缺失或不匹配时报错，不生成新的划分。"""
    if not split_path.is_file():
        raise FileNotFoundError(f"数据/路径问题：未找到已保存索引：{split_path}")
    saved_bytes = split_path.read_bytes()
    split = json.loads(saved_bytes)
    expected_metadata = {
        "schema_version": 1,
        "dataset": "CIFAR10",
        "official_split": "train",
        "dataset_size": len(plain_dataset),
        "classes": plain_dataset.classes,
        "targets_sha256": hashlib.sha256(bytes(plain_dataset.targets)).hexdigest(),
        "split_seed": 42,
        "validation_per_class": 1000,
    }
    if any(split.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("数据/划分问题：已有索引的来源或配置不匹配，请核对文件。")
    train_indices, val_indices = split["train_indices"], split["val_indices"]
    check_split(torch.tensor(plain_dataset.targets), train_indices, val_indices)
    return train_indices, val_indices, hashlib.sha256(saved_bytes).hexdigest()


def make_train_loader(dataset):
    # 每组重新创建相同种子的Generator；shuffle不使用增强消耗的全局随机数。
    order_generator = torch.Generator().manual_seed(ORDER_SEED)
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        generator=order_generator, num_workers=0,
    )


def make_model_and_optimizer(initial_state, device):
    model = SimpleCNN().to(device)
    model.load_state_dict(initial_state, strict=True)  # 复制相同数值，各组参数独立。
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, optimizer


def train_one_epoch(model, loader, criterion, optimizer, device, name, epoch):
    model.train()
    total_loss, total_images = 0.0, 0
    for step, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)  # [B, 3, 32, 32] -> [B, 10]
        loss = criterion(outputs, labels)  # logits [B, 10] + 标签 [B] -> 标量
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_images += images.size(0)
        if step == 1 or step % 1000 == 0 or step == len(loader):
            print(
                f"{name} | Epoch {epoch}/{NUM_EPOCHS} | Batch {step}/{len(loader)} | "
                f"Mean training loss so far: {total_loss / total_images:.6f}",
                flush=True,
            )
    return total_loss / total_images


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_group(name, training_data, train_eval_loader, val_loader,
              initial_state, device, output_dir, checkpoint_dir):
    train_loader = make_train_loader(training_data)
    model, optimizer = make_model_and_optimizer(initial_state, device)
    criterion = torch.nn.CrossEntropyLoss()
    # 模型构造后重置；num_workers=0时随机增强使用本进程的torch随机数。
    torch.manual_seed(AUGMENT_SEED)
    result = {
        "group": name,
        "history": [],
        "best_epoch": None,
        "best_val_accuracy": -1.0,
        "best_checkpoint": str(Path("checkpoints") / checkpoint_dir.name / f"{name}_best.pth"),
        "last_checkpoint": str(Path("checkpoints") / checkpoint_dir.name / f"{name}_last.pth"),
    }
    for epoch in range(1, NUM_EPOCHS + 1):
        training_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, name, epoch,
        )
        # 两组都在无随机增强的原训练图片和同一验证集上评估。
        train_loss, train_accuracy, train_correct, train_count = evaluate(
            model, train_eval_loader, criterion, device, f"{name} Train (plain)",
        )
        val_loss, val_accuracy, val_correct, val_count = evaluate(
            model, val_loader, criterion, device, f"{name} Validation",
        )
        result["history"].append({
            "epoch": epoch,
            "mean_training_loss": training_loss,
            "train_evaluation_loss": train_loss,
            "train_accuracy": train_accuracy,  # JSON内保存0~1的比例。
            "train_correct": train_correct,
            "train_images": train_count,
            "val_evaluation_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_correct": val_correct,
            "val_images": val_count,
        })
        # 严格大于才覆盖；验证accuracy相同时保留较早轮。
        if val_accuracy > result["best_val_accuracy"]:
            result["best_val_accuracy"] = val_accuracy
            result["best_epoch"] = epoch
            torch.save(model.state_dict(), checkpoint_dir / f"{name}_best.pth")
        write_json(output_dir / f"{name}_metrics.json", result)
        print(
            f"{name} | Epoch {epoch}/{NUM_EPOCHS} completed | "
            f"Training loss: {training_loss:.6f} | "
            f"Train accuracy (plain): {train_accuracy:.2%} | "
            f"Validation loss: {val_loss:.6f} | Validation accuracy: {val_accuracy:.2%} | "
            f"Best epoch: {result['best_epoch']}",
            flush=True,
        )
    torch.save(model.state_dict(), checkpoint_dir / f"{name}_last.pth")
    result["epochs_completed"] = NUM_EPOCHS
    result["optimizer_steps"] = NUM_EPOCHS * len(train_loader)
    write_json(output_dir / f"{name}_metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="明确启动两组各5轮训练")
    args = parser.parse_args()
    print(
        f"A: ToTensor | B: RandomCrop(32, padding=4) + Flip(p=0.5) + ToTensor\n"
        f"Shared: SimpleCNN, CrossEntropyLoss, Adam(lr={LEARNING_RATE}), "
        f"batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}\n"
        f"Seeds: initialization={INIT_SEED}, sample order={ORDER_SEED}, augmentation={AUGMENT_SEED}\n"
        "Selection: highest validation accuracy; ties keep the earlier epoch.",
        flush=True,
    )
    if not args.train:
        print("Training not started. Default mode prints config only; add --train explicitly to train.")
        return

    project_dir = Path(__file__).resolve().parent
    plain_data = datasets.CIFAR10(
        root=str(project_dir / "data"), train=True, download=False,
        transform=transforms.ToTensor(),
    )
    augmented_data = datasets.CIFAR10(
        root=str(project_dir / "data"), train=True, download=False,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]),
    )
    split_path = project_dir / "outputs" / "stage9_split_seed42.json"
    train_indices, val_indices, split_sha256 = load_saved_split(plain_data, split_path)
    train_plain = Subset(plain_data, train_indices)
    train_augmented = Subset(augmented_data, train_indices)
    validation = Subset(plain_data, val_indices)
    # 评估只按索引顺序读取；独立Generator避免迭代器消耗增强的随机数。
    train_eval_loader = DataLoader(
        train_plain, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(45),
    )
    val_loader = DataLoader(
        validation, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(46),
    )

    torch.manual_seed(INIT_SEED)
    initial_state = {
        name: value.detach().clone() for name, value in SimpleCNN().state_dict().items()
    }
    initial_sha256 = hashlib.sha256(
        b"".join(value.numpy().tobytes() for value in initial_state.values())
    ).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started_at = datetime.now().astimezone()
    run_name = f"stage9_ab_{started_at.strftime('%Y%m%d_%H%M%S_%f')}"
    output_dir = project_dir / "outputs" / run_name
    checkpoint_dir = project_dir / "checkpoints" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", {
        "run_name": run_name,
        "started_at": started_at.isoformat(),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "split_file": str(split_path.relative_to(project_dir)),
        "split_sha256": split_sha256,
        "split_seed": 42,
        "train_images": len(train_plain),
        "validation_images": len(validation),
        "model": "SimpleCNN",
        "initial_parameters_sha256": initial_sha256,
        "initialization_seed": INIT_SEED,
        "sample_order_seed": ORDER_SEED,
        "augmentation_seed": AUGMENT_SEED,
        "loss": "CrossEntropyLoss",
        "optimizer": "Adam (new for each group)",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "num_workers": 0,
        "batches_per_epoch": len(make_train_loader(train_plain)),
        "augmentation": {
            "A_no_aug": "ToTensor",
            "B_crop_flip": "RandomCrop(32, padding=4) + RandomHorizontalFlip(p=0.5) + ToTensor",
        },
        "normalization": None,
        "selection_rule": "Highest validation accuracy; ties keep the earlier epoch",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "reproducibility": "Seeds control random choices; no cross-device/version bitwise guarantee",
        "scope": "Validation comparison only; no official test-set evaluation",
    })
    print(f"Device: {device} | Train: {len(train_plain)} | Validation: {len(validation)}")
    print("Split reused from:", split_path)
    print("Results directory:", output_dir, flush=True)
    results = {}
    for name, training_data in (("A_no_aug", train_plain), ("B_crop_flip", train_augmented)):
        results[name] = run_group(
            name, training_data, train_eval_loader, val_loader,
            initial_state, device, output_dir, checkpoint_dir,
        )
    comparison = {
        "selection_rule": "Highest validation accuracy; ties keep the earlier epoch",
        "groups": {
            name: {
                "best_epoch": result["best_epoch"],
                "best_val_accuracy": result["best_val_accuracy"],
                "final_epoch": result["history"][-1],
                "best_checkpoint": result["best_checkpoint"],
                "last_checkpoint": result["last_checkpoint"],
            } for name, result in results.items()
        },
        "B_minus_A_best_validation_percentage_points": 100 * (
            results["B_crop_flip"]["best_val_accuracy"] - results["A_no_aug"]["best_val_accuracy"]
        ),
    }
    write_json(output_dir / "comparison.json", comparison)
    print("A/B validation comparison saved to:", output_dir / "comparison.json")


if __name__ == "__main__":
    main()
