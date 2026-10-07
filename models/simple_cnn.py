"""阶段3：建立 CIFAR-10 的第一个 CNN；本文件不读取数据、不启动训练。"""

from torch import nn


class SimpleCNN(nn.Module):
    """输入 [B, 3, 32, 32]；输出 [B, 10] 的类别分数。"""

    def __init__(self):
        super().__init__()

        # Sequential 按下面的顺序执行各层。B 表示 batch 内的图片数量。
        self.features = nn.Sequential(
            # [B, 3, 32, 32] -> [B, 16, 32, 32]
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # 把负数变成 0，shape 不变。
            # [B, 16, 32, 32] -> [B, 16, 16, 16]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [B, 16, 16, 16] -> [B, 32, 16, 16]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # [B, 32, 16, 16] -> [B, 32, 8, 8]
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 保留第 0 维 B，只展开每张图片的通道、高度和宽度。
        self.flatten = nn.Flatten(start_dim=1)  # -> [B, 32 * 8 * 8]
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),  # [B, 2048] -> [B, 64]
            nn.ReLU(),
            nn.Linear(64, 10),  # [B, 64] -> [B, 10]
        )

    def forward(self, x):
        """定义输入经过哪些层；调用 model(images) 时才执行。"""
        x = self.features(x)    # [B, 3, 32, 32] -> [B, 32, 8, 8]
        x = self.flatten(x)     # [B, 32, 8, 8] -> [B, 2048]
        x = self.classifier(x)  # [B, 2048] -> [B, 10]
        return x
