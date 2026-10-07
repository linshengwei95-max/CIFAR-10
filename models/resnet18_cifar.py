"""阶段10：适配 CIFAR-10 的 ResNet18；本文件不读取数据或启动训练。"""

from torch import nn
from torchvision.models import resnet18


def build_resnet18_cifar(num_classes=10):
    """随机初始化；输入 [B, 3, 32, 32]，输出 [B, num_classes] logits。"""
    model = resnet18(weights=None)  # 不加载或下载 ImageNet 预训练权重。

    # 小图片的入口：[B, 3, 32, 32] -> [B, 64, 32, 32]。
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False,
    )
    model.maxpool = nn.Identity()  # 原样传递，避免入口再次缩小高宽。

    # 全局平均池化并展开后，每张图片有512个特征。
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
