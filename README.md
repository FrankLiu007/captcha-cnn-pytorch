# pytorch-captcha-cnn

用 PyTorch + 卷积神经网络从零实现验证码识别，**CPU 和 GPU 都能跑**。

数据是「现生成现用」的：每次取样本都现场画一张新的验证码，
所以训练集理论上无限，也不存在过拟合到固定图片的问题。

参考项目：[ypwhs/captcha_break](https://github.com/ypwhs/captcha_break)（Keras 版多输出 CNN）。
本项目把它的思路用 PyTorch 重写，并把验证码生成器也自己实现了一遍，
方便课堂上逐行讲解。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt
# 需要 GPU 的话装 CUDA 版：
# pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. 训练（自动选择 CPU / GPU）
python train.py

# 3. 看训练好的模型识别效果
python predict.py --num 12

# 4. 算一下准确率
python evaluate.py --batches 20
```

## 常用参数

```bash
python train.py --device cpu                    # 强制用 CPU
python train.py --device cuda                   # 强制用 GPU
python train.py --epochs 2 --steps-per-epoch 20 # 快速跑一遍，验证流程通不通
python train.py --no-refine                    # 只跑第一阶段
python train.py --resume checkpoints/captcha_cnn_best.pt  # 接着上次继续练
```

## 用 uv 管理环境（可选）

项目自带 `pyproject.toml`，也可以用 [uv](https://docs.astral.sh/uv/) 管理虚拟环境：

```bash
uv sync                 # 创建 .venv 并安装依赖
uv run python train.py # 在虚拟环境里跑训练
```

注意：uv 默认从 PyPI 装 PyTorch，那是 **CPU 版**。
要用显卡的话，在 `pyproject.toml` 里加上 CUDA 索引：

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

或者更简单，直接往 `.venv` 里装 CUDA 版：

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## 目录结构

```
pytorch-captcha-cnn/
├── captcha_gen.py      # 验证码生成器（纯 Pillow + numpy，不依赖 torch）
├── dataset.py          # 把生成器包装成 PyTorch Dataset / DataLoader
├── model.py            # CNN 模型
├── train.py            # 训练脚本
├── predict.py          # 识别并画对比图
├── evaluate.py         # 算整体准确率和每一位的准确率
├── plotting.py         # matplotlib 中文字体设置
├── fonts/              # 可选：自带的字体文件
├── samples/            # 生成的样本图（git 忽略）
├── figures/            # 训练曲线、预测样例（git 忽略）
└── checkpoints/        # 模型权重和训练日志（git 忽略）
```

## 模型结构

```
输入 (B, 3, 64, 128)
  │
  ├─ Block1: Conv3x3(32) ×2 + BN + ReLU + MaxPool   → (B,  32, 32, 64)
  ├─ Block2: Conv3x3(64) ×2 + BN + ReLU + MaxPool   → (B,  64, 16, 32)
  ├─ Block3: Conv3x3(128) ×2 + BN + ReLU + MaxPool  → (B, 128,  8, 16)
  ├─ Block4: Conv3x3(256) ×2 + BN + ReLU + MaxPool  → (B, 256,  4,  8)
  ├─ Block5: Conv3x3(256) ×2 + BN + ReLU + MaxPool  → (B, 256,  2,  4)
  │
  ├─ Flatten → (B, 2048)
  ├─ Linear(2048, 512) + ReLU + Dropout
  │
  └─ 4 个独立的分类头 Linear(512, 36)
       ├─ 头1 → 第 1 位字符的 36 类概率
       ├─ 头2 → 第 2 位字符的 36 类概率
       ├─ 头3 → 第 3 位字符的 36 类概率
       └─ 头4 → 第 4 位字符的 36 类概率
```

关键点：卷积部分提取的特征是所有字符**共享**的（笔画、轮廓这些确实通用），
但「这一位是哪个字符」要各自独立判断，所以最后分成 4 个分类头。

这样天然得到两个指标：

- **单字符准确率**：每一位分别统计
- **整体准确率**：4 位全对才算这张验证码答对（更有意义）

损失函数是 4 个交叉熵之和。

## 训练策略

分两个阶段，这是原参考项目里用过的技巧，效果比一直用一个学习率好：

1. **阶段 1**：学习率 `1e-3`，快速收敛
2. **阶段 2**：学习率降到 `1e-4`，精修

每个阶段都有 EarlyStopping（验证集准确率连续 8 轮不提升就停），
始终保存验证准确率最好的那份权重。

## 参考

- [ypwhs/captcha_break](https://github.com/ypwhs/captcha_break) — 原始 Keras 实现
- PyTorch 官方文档：<https://pytorch.org/docs/stable/index.html>
