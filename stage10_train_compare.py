"""阶段10：同条件比较SimpleCNN与ResNet18；默认只显示配置，--train才训练。"""

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import sys
from time import perf_counter

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.simple_cnn import SimpleCNN
from models.resnet18_cifar import build_resnet18_cifar
from stage6_evaluate import evaluate
from stage9_train_ab import load_saved_split, write_json


BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
INIT_SEED = 42
ORDER_SEED = 43


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one_epoch(model, loader, criterion, optimizer, device, name, epoch):
    model.train()  # ResNet18的BatchNorm使用训练模式，并更新运行统计。
    total_loss, total_images, steps = 0.0, 0, 0
    for step, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)  # [B, 3, 32, 32] -> [B, 10]。
        loss = criterion(outputs, labels)  # logits [B, 10] + 标签 [B] -> 标量。
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_images += images.size(0)
        steps += 1
        if step == 1 or step % 1000 == 0 or step == len(loader):
            print(
                f"{name} | Epoch {epoch}/{NUM_EPOCHS} | Batch {step}/{len(loader)} | "
                f"Mean training loss so far: {total_loss / total_images:.6f}",
                flush=True,
            )
    return total_loss / total_images, total_images, steps


def run_group(name, model_factory, train_data, train_eval_loader, val_loader,
              device, output_dir, checkpoint_dir):
    # 结构不同：固定各自初始化种子，但不复制或声称初始参数相同。
    torch.manual_seed(INIT_SEED)
    model = model_factory()
    initial_hash = hashlib.sha256(
        b"".join(value.numpy().tobytes() for value in model.state_dict().values())
    ).hexdigest()
    parameter_count = sum(p.numel() for p in model.parameters())
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    # 每组独立创建相同种子的Generator，保证逐轮原图顺序相同。
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(ORDER_SEED),
    )
    result = {
        "model": name, "parameter_count": parameter_count,
        "initial_state_sha256": initial_hash, "history": [],
        "best_epoch": None, "best_val_accuracy": -1.0,
        "epochs_completed": 0, "optimizer_steps": 0,
        "best_checkpoint": str(Path("checkpoints") / checkpoint_dir.name / f"{name}_best.pth"),
        "last_checkpoint": str(Path("checkpoints") / checkpoint_dir.name / f"{name}_last.pth"),
    }
    for epoch in range(1, NUM_EPOCHS + 1):
        # 先等之前GPU工作完成；计时仅包住本轮训练，结束也等待GPU完成。
        synchronize(device)
        started = perf_counter()
        training_loss, training_images, training_steps = train_one_epoch(
            model, train_loader, criterion, optimizer, device, name, epoch,
        )
        synchronize(device)
        training_seconds = perf_counter() - started

        # 计时已结束；evaluate会切到eval/no_grad，不更新BatchNorm统计。
        train_loss, train_accuracy, train_correct, train_count = evaluate(
            model, train_eval_loader, criterion, device, f"{name} Train (plain)",
        )
        val_loss, val_accuracy, val_correct, val_count = evaluate(
            model, val_loader, criterion, device, f"{name} Validation",
        )
        result["history"].append({
            "epoch": epoch, "mean_training_loss": training_loss,
            "training_images": training_images, "training_steps": training_steps,
            "training_seconds": training_seconds,
            "train_evaluation_loss": train_loss, "train_accuracy": train_accuracy,
            "train_correct": train_correct, "train_images": train_count,
            "val_evaluation_loss": val_loss, "val_accuracy": val_accuracy,
            "val_correct": val_correct, "val_images": val_count,
        })
        result["epochs_completed"] = epoch
        result["optimizer_steps"] += training_steps
        # 严格提高才保存best；验证accuracy同分保留较早轮。
        if val_accuracy > result["best_val_accuracy"]:
            result["best_val_accuracy"] = val_accuracy
            result["best_epoch"] = epoch
            torch.save(model.state_dict(), checkpoint_dir / f"{name}_best.pth")
        write_json(output_dir / f"{name}_metrics.json", result)
        print(
            f"{name} | Epoch {epoch}/{NUM_EPOCHS} completed | "
            f"Training loss: {training_loss:.6f} | Training seconds: {training_seconds:.3f} | "
            f"Train accuracy: {train_accuracy:.2%} | Validation loss: {val_loss:.6f} | "
            f"Validation accuracy: {val_accuracy:.2%} | Best epoch: {result['best_epoch']}",
            flush=True,
        )
    torch.save(model.state_dict(), checkpoint_dir / f"{name}_last.pth")
    result["total_training_seconds"] = sum(row["training_seconds"] for row in result["history"])
    result["mean_epoch_training_seconds"] = result["total_training_seconds"] / len(result["history"])
    write_json(output_dir / f"{name}_metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="明确启动两种新初始化模型各5轮训练")
    args = parser.parse_args()
    print(
        "Models: SimpleCNN, CIFAR ResNet18 (random initialization for each)\n"
        f"Shared: saved 40000/10000 split, ToTensor, Adam(lr={LEARNING_RATE}), "
        f"batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}\n"
        f"Seeds: initialization={INIT_SEED}, sample order={ORDER_SEED}\n"
        "Selection: highest validation accuracy; ties keep the earlier epoch.\n"
        "Timing: training loop only; evaluation and file saving excluded.",
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
    split_path = project_dir / "outputs" / "stage9_split_seed42.json"
    train_indices, val_indices, split_hash = load_saved_split(plain_data, split_path)
    train_data, val_data = Subset(plain_data, train_indices), Subset(plain_data, val_indices)
    train_eval_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(45),
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(46),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started_at = datetime.now().astimezone()
    run_name = f"stage10_compare_{started_at.strftime('%Y%m%d_%H%M%S_%f')}"
    output_dir, checkpoint_dir = project_dir / "outputs" / run_name, project_dir / "checkpoints" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", {
        "run_name": run_name, "started_at": started_at.isoformat(),
        "python": sys.executable, "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__, "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "models_in_run_order": ["SimpleCNN", "CIFAR_ResNet18"],
        "split_file": str(split_path.relative_to(project_dir)), "split_sha256": split_hash,
        "train_images": len(train_data), "validation_images": len(val_data),
        "initialization_seed": INIT_SEED, "sample_order_seed": ORDER_SEED,
        "initialization_note": "Different architectures have independent random parameters; values are not identical",
        "loss": "CrossEntropyLoss", "optimizer": "Adam (new for each model)",
        "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS,
        "batches_per_epoch": (len(train_data) + BATCH_SIZE - 1) // BATCH_SIZE,
        "num_workers": 0, "transform": "ToTensor", "augmentation": None, "normalization": None,
        "selection_rule": "Highest validation accuracy; ties keep the earlier epoch",
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "timing_scope": "End-to-end training loop including data, transfers, updates and progress printing",
        "timing_excludes": "Dataset/model setup, training/validation evaluation, checkpoint and JSON saving",
        "timing_note": "CUDA synchronized at boundaries; first epoch included; single sequential run, not an inference benchmark",
        "scope": "New model comparison; no official test-set evaluation or reuse of trained weights",
    })
    print(f"Device: {device} | Train: {len(train_data)} | Validation: {len(val_data)}")
    print("Split reused from:", split_path)
    print("Results directory:", output_dir, flush=True)
    results = {}
    for name, factory in (("SimpleCNN", SimpleCNN), ("CIFAR_ResNet18", build_resnet18_cifar)):
        results[name] = run_group(
            name, factory, train_data, train_eval_loader, val_loader,
            device, output_dir, checkpoint_dir,
        )
    simple, resnet = results["SimpleCNN"], results["CIFAR_ResNet18"]
    comparison = {
        "selection_rule": "Highest validation accuracy; ties keep the earlier epoch",
        "groups": {
            name: {key: result[key] for key in (
                "parameter_count", "best_epoch", "best_val_accuracy", "epochs_completed",
                "optimizer_steps", "total_training_seconds", "mean_epoch_training_seconds",
                "best_checkpoint", "last_checkpoint",
            )} for name, result in results.items()
        },
        "ResNet18_minus_SimpleCNN_best_validation_percentage_points": 100 * (
            resnet["best_val_accuracy"] - simple["best_val_accuracy"]
        ),
        "ResNet18_over_SimpleCNN_mean_epoch_training_seconds": (
            resnet["mean_epoch_training_seconds"] / simple["mean_epoch_training_seconds"]
        ),
        "timing_scope": "Training loop only; first epoch included; evaluation and saving excluded",
    }
    write_json(output_dir / "comparison.json", comparison)
    print("Model validation/time comparison saved to:", output_dir / "comparison.json")


if __name__ == "__main__":
    main()
