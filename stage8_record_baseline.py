"""阶段8：保存2026-10-07已完成的基线实验记录，不执行训练或评估。"""

from pathlib import Path


def main():
    # 数值来自已核对的阶段5/6源码和日志，不是重新运行得到的结果。
    config = {
        "实验编号": "baseline_01",
        "实验日期": "2026-10-07（Asia/Shanghai）",
        "模型": "SimpleCNN，136,874个可学习参数",
        "数据": "CIFAR-10：官方训练集50,000张、测试集10,000张",
        "独立验证集": "本次未划分",
        "损失函数": "CrossEntropyLoss",
        "优化器": "Adam",
        "learning rate": 0.001,
        "batch size": 8,
        "epoch": 5,
        "数据预处理": "ToTensor()；未使用Normalize",
        "数据增强": "无",
        "训练shuffle / num_workers": "True / 0",
        "评估shuffle / batch size / num_workers": "False / 8 / 0",
        "随机种子": "未固定",
        "训练设备": "CUDA，NVIDIA GeForce RTX 5060 Laptop GPU",
    }
    results = {
        "每轮batch数 / 总参数更新次数": "6,250 / 31,250",
        "第5轮平均training loss": 0.870657,
        "训练集accuracy": "71.73%（35,863 / 50,000；精确值71.726%）",
        "训练集mean evaluation loss": 0.800134,
        "测试集accuracy": "65.93%（6,593 / 10,000）",
        "测试集mean evaluation loss": 0.981316,
        "train-test accuracy差距": "约5.80个百分点",
    }

    sections = ["# CIFAR-10 基线实验记录\n"]
    for title, values in (("实验配置", config), ("实验结果", results)):
        rows = [f"| {key} | {value} |" for key, value in values.items()]
        sections.append(
            f"## {title}\n\n| 项目 | 记录 |\n| --- | --- |\n" + "\n".join(rows)
        )
    sections.append("""## 来源与保存文件

- 模型：[simple_cnn.py](../models/simple_cnn.py)；配置：[stage5_train.py](../stage5_train.py)。
- 训练输出：[5轮完整日志](stage5_five_epochs_run_2026-10-07.txt)。
- 评估输出：[训练/测试评估日志](stage6_evaluation_run_2026-10-07.txt)；代码：[stage6_evaluate.py](../stage6_evaluate.py)。
- 曲线：[5轮平均training loss](stage7_training_loss.png)。
- 权重：[stage5_last.pth](../checkpoints/stage5_last.pth)，第5轮最后模型的state_dict；未按验证集选择最好模型，未保存optimizer状态。

## 结果含义与比较范围

- 本表整理学生已有运行结果；记录脚本只写文档，不计算新指标。修改表内数值不会改变实际训练结果。
- 第5轮training loss在逐批更新过程中统计；evaluation loss来自固定的第5轮模型，两者不要求相等。
- 当前仅有第5轮模型的accuracy，没有逐轮accuracy记录；仅凭训练loss下降或本次train-test差距，不能确认过拟合。
- 训练见过全部50,000张官方训练图片。正式调参前先划分独立训练/验证集，再训练新模型；用验证集比较和选择，测试集留作最终评估。本次测试结果作为已完成基线的记录，不作为后续模型选择依据。
- 本次未固定随机种子；重跑数值可能不同，不能据一次小幅差异认定某项改动有效。
- 环境沿用此前已验证记录：Python 3.10.20、torch 2.13.0+cu130、torchvision 0.28.0+cpu、matplotlib 3.10.9；本记录脚本未重新查询或改动环境。
""")

    output_path = Path(__file__).resolve().parent / "outputs" / "stage8_baseline.md"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text("\n\n".join(sections), encoding="utf-8")
    print("Baseline record saved to:", output_path)


if __name__ == "__main__":
    main()
