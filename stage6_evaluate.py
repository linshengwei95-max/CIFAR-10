"""阶段6：加载阶段5最后参数，评估训练集和测试集，不更新参数。"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.simple_cnn import SimpleCNN


def evaluate(model, data_loader, criterion, device, split_name):
    """返回固定模型的平均loss、准确率、正确图片数和已评估图片数。"""
    model.eval()  # 设置评估模式；这一步本身不关闭梯度记录。
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    with torch.no_grad():  # 评估forward不需要记录反向传播所需的信息。
        for step, (images, labels) in enumerate(data_loader, start=1):
            images = images.to(device)  # [B, 3, 32, 32]
            labels = labels.to(device)  # [B]
            outputs = model(images)  # [B, 10]，原始logits
            loss = criterion(outputs, labels)  # 本批平均loss，标量

            predicted_labels = outputs.argmax(dim=1)  # [B]
            # 比较得到[B]的布尔Tensor，sum统计这一批分对了多少张。
            total_correct += (predicted_labels == labels).sum().item()
            total_images += images.size(0)
            total_loss += loss.item() * images.size(0)

            if step == 1 or step % 1000 == 0 or step == len(data_loader):
                print(
                    f"{split_name} evaluation | "
                    f"Batch {step}/{len(data_loader)} | "
                    f"Images evaluated: {total_images}",
                    flush=True,
                )

    return (
        total_loss / total_images,
        total_correct / total_images,
        total_correct,
        total_images,
    )


def main():
    project_dir = Path(__file__).resolve().parent
    checkpoint_path = project_dir / "checkpoints" / "stage5_last.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"未找到阶段5的权重，请核对文件路径：{checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    saved_parameters = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True,
    )
    model.load_state_dict(saved_parameters, strict=True)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    batch_size = 8

    print("Device:", device)
    print("Loaded model parameters from:", checkpoint_path)
    print("Evaluation batch_size:", batch_size)

    # CIFAR10的train参数选择哪份数据；它不控制是否更新模型参数。
    for split_name, use_training_set in (("Train", True), ("Test", False)):
        dataset = datasets.CIFAR10(
            root=str(project_dir / "data"), train=use_training_set,
            download=False, transform=transforms.ToTensor(),
        )
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        )
        mean_loss, accuracy, correct_count, image_count = evaluate(
            model, data_loader, criterion, device, split_name,
        )
        print(f"{split_name} images evaluated: {image_count}")
        print(f"{split_name} correct predictions: {correct_count}")
        print(f"{split_name} mean evaluation loss: {mean_loss:.6f}")
        print(f"{split_name} accuracy: {accuracy:.2%}", flush=True)


if __name__ == "__main__":
    main()
