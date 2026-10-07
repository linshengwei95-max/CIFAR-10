"""补充评估：固定阶段12最佳ResNet18；--evaluate才评估官方测试集。"""

import argparse
import hashlib
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter


SOURCE_RUN = "stage12_lr_20261007_195820_358131"
GROUP = "B_lr_0p0003"
BATCH_SIZE = 8


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class Tee:
    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def run_evaluation(project_dir, output_dir, started_at):
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from models.resnet18_cifar import build_resnet18_cifar
    from stage6_evaluate import evaluate

    source_dir = project_dir / "outputs" / SOURCE_RUN
    config_path = source_dir / "config.json"
    comparison_path = source_dir / "comparison.json"
    metrics_path = source_dir / f"{GROUP}_metrics.json"
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected = comparison["groups"][GROUP]
    if selected["best_val_accuracy"] != max(
        group["best_val_accuracy"] for group in comparison["groups"].values()
    ):
        raise ValueError("固定选择的组不再是源对照中验证准确率最高的组。")
    if (metrics["best_epoch"], metrics["best_val_accuracy"]) != (
        selected["best_epoch"], selected["best_val_accuracy"]
    ):
        raise ValueError("源指标和comparison中的最佳轮次或验证准确率不一致。")
    if (source_config["model"] != "CIFAR_ResNet18"
            or source_config["transform"] != "ToTensor"
            or source_config["augmentation"] is not None
            or source_config["normalization"] is not None):
        raise ValueError("源实验的模型或预处理已改变，请先核对。")

    checkpoint_path = project_dir / "checkpoints" / SOURCE_RUN / f"{GROUP}_best.pth"
    split_path = project_dir / "outputs" / "stage9_split_seed42.json"
    protected_paths = [
        *sorted(project_dir.glob("stage*.py")),
        *sorted((project_dir / "models").glob("*.py")),
        config_path, comparison_path, metrics_path, split_path, checkpoint_path,
        checkpoint_path.with_name(f"{GROUP}_last.pth"),
        *sorted(path for path in (project_dir / "data" / "cifar-10-batches-py").iterdir()
                if path.is_file()),
    ]
    before_hashes = {
        path.relative_to(project_dir).as_posix(): sha256(path)
        for path in protected_paths
    }
    if before_hashes[split_path.relative_to(project_dir).as_posix()] != source_config["split_sha256"]:
        raise ValueError("源实验的训练/验证划分索引校验值不一致。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = source_config["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = source_config["cudnn_deterministic"]
    saved_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not all(torch.isfinite(value).all().item() for value in saved_state.values()):
        raise ValueError("已有权重包含非有限值，停止评估。")
    model = build_resnet18_cifar()
    model.load_state_dict(saved_state, strict=True)
    model = model.to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != selected["parameter_count"]:
        raise ValueError("实际模型参数量与源记录不一致。")

    # train=False选择官方测试图片；它与model.eval()控制不同的事情。
    test_data = datasets.CIFAR10(
        root=str(project_dir / "data"), train=False, download=False,
        transform=transforms.ToTensor(),
    )
    if len(test_data) != 10000:
        raise ValueError("官方测试集图片数量应为10000。")
    loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    criterion = torch.nn.CrossEntropyLoss()
    source_epoch = next(item for item in metrics["history"]
                        if item["epoch"] == selected["best_epoch"])
    config = {
        "run_name": output_dir.name, "started_at": started_at.isoformat(),
        "source_run": SOURCE_RUN, "group": GROUP,
        "model": "CIFAR_ResNet18", "training_learning_rate": selected["learning_rate"],
        "selected_epoch": selected["best_epoch"],
        "selection_rule": "Fixed before test evaluation; highest recorded validation accuracy",
        "source_validation_accuracy": selected["best_val_accuracy"],
        "checkpoint": checkpoint_path.relative_to(project_dir).as_posix(),
        "checkpoint_sha256": before_hashes[checkpoint_path.relative_to(project_dir).as_posix()],
        "dataset": "CIFAR-10 official test", "dataset_train": False,
        "test_images": len(test_data), "transform": "ToTensor",
        "augmentation": None, "normalization": None,
        "batch_size": BATCH_SIZE, "batches": len(loader),
        "shuffle": False, "num_workers": 0,
        "loss": "CrossEntropyLoss", "python": sys.executable,
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "scope": "One evaluation of the selected existing checkpoint; no training or test-based selection",
    }
    save_json(output_dir / "config.json", config)
    print("Device:", config["device"], "|", config["device_name"])
    print("Source run:", SOURCE_RUN, "| Group:", GROUP)
    print("Loaded checkpoint:", checkpoint_path)
    print("Selected epoch:", selected["best_epoch"])
    print(f"Recorded validation accuracy: {selected['best_val_accuracy']:.2%}")
    print("Official test images:", len(test_data), "| Batches:", len(loader))

    first_batch = {}
    observed_batches = 0

    # 观察实际评估的首批shape，不另做一次前向计算。
    def observe_forward(module, inputs, outputs):
        nonlocal observed_batches
        observed_batches += 1
        if not first_batch:
            first_batch.update({
                "images_shape": list(inputs[0].shape),
                "logits_shape": list(outputs.shape),
                "model_training": module.training,
                "grad_enabled": torch.is_grad_enabled(),
                "logits_requires_grad": outputs.requires_grad,
            })

    def observe_labels(module, inputs):
        if "labels_shape" not in first_batch:
            first_batch["labels_shape"] = list(inputs[1].shape)
            first_batch["labels_dtype"] = str(inputs[1].dtype)

    forward_handle = model.register_forward_hook(observe_forward)
    label_handle = criterion.register_forward_pre_hook(observe_labels)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    try:
        # 复用函数内部执行model.eval()、no_grad；没有backward或step。
        mean_loss, accuracy, correct, images = evaluate(
            model, loader, criterion, device, "Official test",
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        forward_handle.remove()
        label_handle.remove()
    elapsed = perf_counter() - started

    # 同时核对全部参数与BatchNorm缓冲区，确认评估没有更新模型状态。
    state_unchanged = all(
        torch.equal(value.detach().cpu(), saved_state[name])
        for name, value in model.state_dict().items()
    )
    after_hashes = {
        path.relative_to(project_dir).as_posix(): sha256(path)
        for path in protected_paths
    }
    if not state_unchanged or before_hashes != after_hashes:
        raise RuntimeError("评估前后模型状态或受保护文件发生变化，请核对。")
    if images != len(test_data) or observed_batches != len(loader):
        raise RuntimeError("实际评估样本数或前向批次数不完整。")
    expected_first = {
        "images_shape": [8, 3, 32, 32], "logits_shape": [8, 10],
        "model_training": False, "grad_enabled": False,
        "logits_requires_grad": False, "labels_shape": [8],
        "labels_dtype": "torch.int64",
    }
    if first_batch != expected_first:
        raise RuntimeError(f"实际首批shape或评估模式异常：{first_batch}")

    result = {
        "source_run": SOURCE_RUN, "group": GROUP, "selected_epoch": selected["best_epoch"],
        "test_loss": mean_loss, "test_accuracy": accuracy,
        "test_correct": correct, "test_errors": images - correct,
        "test_images": images, "batches_evaluated": observed_batches,
        "evaluation_seconds": elapsed,
        "source_validation_accuracy": selected["best_val_accuracy"],
        "source_validation_loss": source_epoch["val_evaluation_loss"],
        "test_minus_validation_percentage_points": 100 * (accuracy - selected["best_val_accuracy"]),
        "first_batch": first_batch, "strict_checkpoint_load": True,
        "checkpoint_tensors_finite": True, "model_state_unchanged": state_unchanged,
        "protected_files_unchanged": before_hashes == after_hashes,
        "scope": config["scope"],
    }
    save_json(output_dir / "result.json", result)
    save_json(output_dir / "verification.json", {
        "before_sha256": before_hashes, "after_sha256": after_hashes,
        "model_state_unchanged": state_unchanged,
        "first_batch": first_batch, "batches_evaluated": observed_batches,
        "test_images": images,
    })
    summary = (
        "# 当前最佳模型的官方测试集评估\n\n"
        f"评估日期：{started_at:%Y-%m-%d}（Asia/Shanghai）。\n\n"
        f"固定选择 `{SOURCE_RUN}` 的 `{GROUP}_best.pth`：适配ResNet18，"
        f"训练学习率{selected['learning_rate']}，最佳第{selected['best_epoch']}轮。"
        "选择依据是已有验证准确率；本次测试结果未参与模型或超参数选择。\n\n"
        "| 数据 | 图片数 | 正确数 | 准确率 | 平均loss |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
        f"| 源实验验证集 | {source_epoch['val_images']:,} | {source_epoch['val_correct']:,} | "
        f"{selected['best_val_accuracy']:.2%} | {source_epoch['val_evaluation_loss']:.6f} |\n"
        f"| 本次官方测试集 | {images:,} | {correct:,} | {accuracy:.2%} | {mean_loss:.6f} |\n\n"
        f"测试减验证准确率为{result['test_minus_validation_percentage_points']:+.2f}个百分点。"
        "两份数据包含不同图片，这个差异描述本次结果，不能单凭它诊断原因。\n\n"
        f"本次在{config['device_name']}上完成{observed_batches:,}批，"
        f"评估循环耗时{elapsed:.2f}秒。沿用ToTensor、batch=8、shuffle=False；"
        "复用stage6的evaluate，以eval/no_grad完成一次完整测试集评估。"
        "首批图片[8,3,32,32]、标签[8]、logits[8,10]。\n\n"
        "严格权重加载、全部权重Tensor有限性、实际样本数/批次数和首批shape核对通过；"
        "模型全部参数及BatchNorm缓冲区在评估前后逐Tensor一致，受保护文件SHA256未变。"
        "没有训练、参数更新、安装升级或重复种子实验。\n\n"
        "本结果只描述所选既有模型的测试表现。早期65.93%测试基线的模型和训练规模不同，"
        "不把二者差异归因于单个变量；学习率对照结论仍来自已有验证比较。"
        "\n\n"
        "记录：[配置](config.json) · [结果](result.json) · "
        "[完整运行输出](console_output.txt) · [核对记录](verification.json)\n"
    )
    (output_dir / "evaluation_summary.md").write_text(summary, encoding="utf-8")
    print("First batch:", json.dumps(first_batch, ensure_ascii=False))
    print(f"Test correct: {correct}/{images} | Errors: {images - correct}")
    print(f"Test mean loss: {mean_loss:.6f} | Test accuracy: {accuracy:.2%}")
    print(f"Test minus validation: {result['test_minus_validation_percentage_points']:+.2f} percentage points")
    print("Model state and protected files unchanged: True")
    print("Results saved to:", output_dir, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help="明确评估固定最佳权重的官方测试表现")
    args = parser.parse_args()
    print(f"Plan: {SOURCE_RUN}/{GROUP}_best.pth | official test 10000 | ToTensor | batch={BATCH_SIZE}")
    if not args.evaluate:
        print("Evaluation not started. Add --evaluate explicitly to run.")
        return
    project_dir = Path(__file__).resolve().parent
    started_at = datetime.now(timezone(timedelta(hours=8)))
    output_dir = project_dir / "outputs" / f"stage15_test_{started_at:%Y%m%d_%H%M%S_%f}"
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "console_output.txt").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)):
            print("Run directory:", output_dir, flush=True)
            run_evaluation(project_dir, output_dir, started_at)


if __name__ == "__main__":
    main()
