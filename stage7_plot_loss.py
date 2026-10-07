"""阶段7：绘制已有5轮平均training loss，只读取已记录的数值。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    # 来源：outputs/stage5_five_epochs_run_2026-10-07.txt的5行轮末均值。
    # 保留日志显示的6位小数；每个点是该轮训练过程中的平均loss。
    epochs = [1, 2, 3, 4, 5]
    mean_training_losses = [1.485624, 1.123879, 0.996464, 0.923083, 0.870657]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, mean_training_losses, marker="o", linewidth=2, color="#2563eb")
    ax.set_title("CIFAR-10: mean training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean training loss")
    ax.set_xticks(epochs)
    ax.set_xlim(0.75, 5.25)
    ax.set_ylim(0, max(mean_training_losses) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    for epoch, mean_loss in zip(epochs, mean_training_losses):
        ax.annotate(
            f"{mean_loss:.6f}", (epoch, mean_loss),
            xytext=(0, 10), textcoords="offset points", ha="center",
        )

    fig.tight_layout()
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    image_path = output_dir / "stage7_training_loss.png"
    fig.savefig(image_path, dpi=160)
    plt.close(fig)

    print("Epochs:", epochs)
    print("Mean training losses:", mean_training_losses)
    print("Training loss plot saved to:", image_path)


if __name__ == "__main__":
    main()
