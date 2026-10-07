"""阶段12：由学生选定学习率做ResNet18对照；--train才启动训练。"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter


BATCH_SIZE = 8
NUM_EPOCHS = 5
INIT_SEED = 42
ORDER_SEED = 43
GROUPS = (("A_lr_0p001", 0.001), ("B_lr_0p0003", 0.0003))
REFERENCE_RUN = "stage10_compare_20261007_181857_108622"


def state_sha256(state):
    """包含参数及BatchNorm统计；仅在学生明确启动训练后调用。"""
    return hashlib.sha256(
        b"".join(value.detach().cpu().numpy().tobytes() for value in state.values())
    ).hexdigest()


def run_group(name, learning_rate, initial_state, initial_hash, model_factory,
              train_data, train_eval_loader, val_loader, device,
              output_dir, checkpoint_dir):
    import torch
    from torch.utils.data import DataLoader
    from stage6_evaluate import evaluate
    from stage10_train_compare import synchronize, train_one_epoch
    from stage9_train_ab import write_json

    model = model_factory()
    model.load_state_dict(initial_state, strict=True)  # 两组复制同一初始数值，参数存储独立。
    actual_initial_hash = state_sha256(model.state_dict())
    if actual_initial_hash != initial_hash:
        raise ValueError("模型/初始化问题：当前组没有从共同初始状态开始。")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 唯一改变的因素。
    criterion = torch.nn.CrossEntropyLoss()

    # 两组都从相同的随机数与图片顺序开始，不沿用A训练后的模型/优化器。
    torch.manual_seed(INIT_SEED)
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(ORDER_SEED),
    )
    result = {
        "group": name, "model": "CIFAR_ResNet18", "learning_rate": learning_rate,
        "parameter_count": parameter_count, "initial_state_sha256": actual_initial_hash,
        "history": [], "best_epoch": None, "best_val_accuracy": -1.0,
        "epochs_completed": 0, "optimizer_steps": 0,
        "best_checkpoint": str(Path("checkpoints") / checkpoint_dir.name / f"{name}_best.pth"),
        "last_checkpoint": str(Path("checkpoints") / checkpoint_dir.name / f"{name}_last.pth"),
    }
    print(f"Starting {name} | Adam learning rate: {learning_rate} | Initial state: {actual_initial_hash}")
    for epoch in range(1, NUM_EPOCHS + 1):
        synchronize(device)
        started = perf_counter()
        training_loss, training_images, training_steps = train_one_epoch(
            model, train_loader, criterion, optimizer, device, name, epoch,
        )
        synchronize(device)
        training_seconds = perf_counter() - started

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
    result["mean_epoch_training_seconds"] = result["total_training_seconds"] / NUM_EPOCHS
    write_json(output_dir / f"{name}_metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="明确启动学习率0.001/0.0003两组各5轮训练")
    args = parser.parse_args()
    print(
        "Experiment: CIFAR ResNet18 learning-rate comparison\n"
        "A: Adam(lr=0.001) | B: Adam(lr=0.0003)\n"
        f"Shared: saved 40000/10000 split, ToTensor, batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}\n"
        f"Seeds: shared initialization={INIT_SEED}, sample order={ORDER_SEED}\n"
        "Both groups copy the same initial parameters and BatchNorm state.\n"
        "Selection: highest validation accuracy; ties keep the earlier epoch.",
        flush=True,
    )
    if not args.train:
        print("Training not started. Add --train explicitly to start the comparison.")
        return

    import torch
    import torchvision
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from models.resnet18_cifar import build_resnet18_cifar
    from stage9_train_ab import load_saved_split, write_json

    project_dir = Path(__file__).resolve().parent
    reference_dir = project_dir / "outputs" / REFERENCE_RUN
    reference_config = json.loads((reference_dir / "config.json").read_text(encoding="utf-8"))
    reference_metrics = json.loads((reference_dir / "CIFAR_ResNet18_metrics.json").read_text(encoding="utf-8"))
    plain_data = datasets.CIFAR10(
        root=str(project_dir / "data"), train=True, download=False,
        transform=transforms.ToTensor(),
    )
    split_path = project_dir / "outputs" / "stage9_split_seed42.json"
    train_indices, val_indices, split_hash = load_saved_split(plain_data, split_path)
    if split_hash != reference_config["split_sha256"]:
        raise ValueError("数据/划分问题：当前固定索引与已有参考实验不一致。")
    train_data = Subset(plain_data, train_indices)
    val_data = Subset(plain_data, val_indices)
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

    torch.manual_seed(INIT_SEED)
    initial_model = build_resnet18_cifar()
    initial_state = {
        key: value.detach().clone() for key, value in initial_model.state_dict().items()
    }
    initial_hash = state_sha256(initial_state)
    del initial_model
    started_at = datetime.now().astimezone()
    run_name = f"stage12_lr_{started_at.strftime('%Y%m%d_%H%M%S_%f')}"
    output_dir = project_dir / "outputs" / run_name
    checkpoint_dir = project_dir / "checkpoints" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "config.json", {
        "run_name": run_name, "started_at": started_at.isoformat(),
        "python": sys.executable, "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__, "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "model": "CIFAR_ResNet18", "variable": "Adam learning rate",
        "groups": [{"name": name, "learning_rate": lr} for name, lr in GROUPS],
        "split_file": str(split_path.relative_to(project_dir)), "split_sha256": split_hash,
        "train_images": len(train_data), "validation_images": len(val_data),
        "batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS, "num_workers": 0,
        "batches_per_epoch": (len(train_data) + BATCH_SIZE - 1) // BATCH_SIZE,
        "initialization_seed": INIT_SEED, "sample_order_seed": ORDER_SEED,
        "shared_initial_state_sha256": initial_hash,
        "initialization_note": "Both groups copy the same new initial parameters and BatchNorm state",
        "optimizer": "Adam (new optimizer for each group)", "loss": "CrossEntropyLoss",
        "transform": "ToTensor", "augmentation": None, "normalization": None,
        "selection_rule": "Highest validation accuracy; ties keep the earlier epoch",
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "historical_reference": {
            "source_run": REFERENCE_RUN, "model": "CIFAR_ResNet18",
            "best_epoch": reference_metrics["best_epoch"],
            "best_val_accuracy": reference_metrics["best_val_accuracy"],
            "scope": "Historical record; the current A and B groups both train afresh",
        },
        "timing_scope": "Training loop only; CUDA synchronized; evaluation and saving excluded",
        "scope": "Student-selected learning-rate comparison; no official test-set use",
    })
    print(f"Device: {device} | Train: {len(train_data)} | Validation: {len(val_data)}")
    print("Shared initial state SHA256:", initial_hash)
    print("Split reused from:", split_path)
    print("Results directory:", output_dir, flush=True)
    results = {}
    for name, learning_rate in GROUPS:
        results[name] = run_group(
            name, learning_rate, initial_state, initial_hash, build_resnet18_cifar,
            train_data, train_eval_loader, val_loader, device, output_dir, checkpoint_dir,
        )
    group_a, group_b = (results[name] for name, _ in GROUPS)
    comparison = {
        "variable": "Adam learning rate",
        "selection_rule": "Highest validation accuracy; ties keep the earlier epoch",
        "shared_initial_state_sha256": initial_hash,
        "groups": {
            name: {key: result[key] for key in (
                "learning_rate", "parameter_count", "initial_state_sha256",
                "best_epoch", "best_val_accuracy", "epochs_completed",
                "optimizer_steps", "total_training_seconds", "mean_epoch_training_seconds",
                "best_checkpoint", "last_checkpoint",
            )} for name, result in results.items()
        },
        "B_minus_A_best_validation_percentage_points": 100 * (
            group_b["best_val_accuracy"] - group_a["best_val_accuracy"]
        ),
        "scope": "One paired comparison under the recorded fixed conditions",
    }
    write_json(output_dir / "comparison.json", comparison)
    print("Learning-rate comparison saved to:", output_dir / "comparison.json")


if __name__ == "__main__":
    main()
