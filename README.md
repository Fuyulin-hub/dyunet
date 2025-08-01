# DyUNet 动态卷积模型实验

本项目包含一个简化版的 DyUNet 模型和基本训练流程，适合初学者学习 U-Net 结构与动态卷积原理。

---

## 📁 文件结构

- `dyunet1.py`：定义了 DyUNet 模型结构（轻量级 U-Net + 动态权重）
- `train.py`：训练脚本，使用假数据进行快速验证

---

## 🚀 如何运行

1. 确保安装了 PyTorch 并有 GPU 支持
2. 运行训练：

```bash
python train.py
