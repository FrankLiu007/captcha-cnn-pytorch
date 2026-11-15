# PyTorch 验证码识别 CNN — 课堂演示项目

用 PyTorch + 卷积神经网络从零实现验证码识别，**CPU 和 GPU 都能跑**。

参考项目：[ypwhs/captcha_break](https://github.com/ypwhs/captcha_break)（Keras 版多输出 CNN）。
本项目把它的思路用 PyTorch 重写，并且把验证码生成器也自己实现了一遍，
方便课堂上逐行讲解「训练数据到底长什么样」。

> **教学目标**
> 1. 看懂一个 CNN 从输入到输出的完整数据流（形状怎么变）
> 2. 理解「多输出」和「多分类」的区别，以及为什么要有 4 个分类头
> 3. 理解「整体准确率」和「单字符准确率」为什么不一样
> 4. 亲手跑一遍训练，观察 loss 下降和准确率上升的过程

---

## 1. 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 需要 GPU 的话装 CUDA 版 PyTorch：
# pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. 训练（自动选择 CPU / GPU）
python train.py

# 3. 看训练好的模型识别效果（会画图）
python predict.py --num 12

# 4. 算准确率
python evaluate.py --batches 20
```

### 常用参数

| 命令 | 作用 |
|---|---|
| `python train.py --device cpu` | 强制用 CPU |
| `python train.py --device cuda` | 强制用 GPU |
| `python train.py --epochs 2 --steps-per-epoch 20` | 快速跑一遍，验证流程通不通 |
| `python train.py --no-refine` | 只跑第一阶段（不做精修） |
| `python train.py --resume checkpoints/captcha_cnn_best.pt` | 接着上次继续练 |
| `python predict.py --text "AB12"` | 指定文本，生成并识别这一张 |
| `python evaluate.py --batches 100` | 多跑一些样本，数字更可信 |

### CPU 和 GPU 的选择

代码里统一用 `--device` 控制，默认 `auto`：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
images = images.to(device)   # 数据也要搬到同一块设备上
```

**这是 PyTorch 最容易踩的坑**：模型和输入数据必须在同一个设备上，
否则会报 `Expected all tensors to be on the same device`。
模型搬家用 `.to(device)`，数据搬家用 `tensor.to(device)`。

---

## 2. 目录结构

```
pytorch-captcha-cnn/
├── captcha_gen.py      # 验证码生成器（纯 Pillow + numpy，不依赖 torch）
├── dataset.py          # 把生成器包装成 PyTorch Dataset / DataLoader
├── model.py            # CNN 模型
├── train.py            # 训练脚本
├── predict.py          # 识别并画对比图
├── evaluate.py         # 算整体准确率和每一位的准确率
├── fonts/              # 可选：自带的字体文件
├── samples/            # 生成的样本图（git 忽略）
├── figures/            # 训练曲线、预测样例（git 忽略）
└── checkpoints/        # 模型权重和训练日志（git 忽略）
```

---

## 3. 各模块说明

### 3.1 `captcha_gen.py` — 验证码生成器

模仿 `ImageCaptcha` 的思路自己实现，步骤：

1. 铺一层浅色背景
2. 画 2~4 条随机颜色的干扰线
3. 随机撒一些噪点
4. 每个字符用随机颜色画上去，带 ±20° 随机旋转和上下错位

默认字符集是 10 个数字 + 26 个大写字母（共 **36 类**），长度 4，尺寸 128×64。

```python
from captcha_gen import CaptchaGenerator

gen = CaptchaGenerator()
image, text = gen.generate()          # 随机文本
image, text = gen.generate("AB12")    # 指定文本

# 批量生成，输出已经转成 PyTorch 要的格式
images, texts = gen.generate_batch(32)
# images 形状 (32, 3, 64, 128)，数值在 [0, 1]
```

**为什么自己写而不用 `captcha` 库？**
`captcha` 库比较老，在新版 Python 上容易装不上；更重要的是它的代码学生看不到。
自己实现一遍，学生能看清验证码是怎么画出来的，也知道训练数据从哪来。

**关键细节：输出的形状是 `(B, 3, H, W)` 而不是 `(B, H, W, 3)`。**
这是 PyTorch 的约定（通道在前），和 Keras / TensorFlow 的约定相反，
第一次接触很容易搞错。转换代码在 `generate_batch()` 里：

```python
array = array.transpose(0, 3, 1, 2) / 255.0   # HWC -> CHW，再归一化
```

### 3.2 `dataset.py` — 数据集

把生成器包成 `Dataset`，每个样本返回 `(图像张量, 标签索引列表)`：

- 图像张量：`float32`，形状 `(3, 64, 128)`，数值在 `[0, 1]`
- 标签：长度为 4 的整数列表，每位字符对应字符集中的下标

**标签为什么不是 one-hot？**
因为每一位要配一个独立的分类头，用交叉熵直接吃整数标签更省事，
不用自己转 one-hot 再转回来。

```python
from dataset import make_dataloaders

train_loader, val_loader = make_dataloaders(
    batch_size=128, train_steps=500, val_batches=40, num_workers=0
)
```

`num_workers` 的说明：数据是现场生成的，生成速度可能成为瓶颈，
可以用多进程并行。**Windows 上建议设 0**（多进程有坑，
而且必须把入口包在 `if __name__ == "__main__"` 里），
Linux / macOS 上可以设 4~8 提速。

### 3.3 `model.py` — 网络结构

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
  ├─ Dropout(0.5)
  │
  └─ 4 个独立的分类头 Linear(2048, 36)
       ├─ 头1 → 第 1 位字符的 36 类概率
       ├─ 头2 → 第 2 位字符的 36 类概率
       ├─ 头3 → 第 3 位字符的 36 类概率
       └─ 头4 → 第 4 位字符的 36 类概率
```

几个值得讲一讲的点：

**（1）为什么是「两层卷积 + 一次池化」？**
在同样的池化次数下，两层 3×3 卷积比一层 3×3 的感受野更大，
效果比一层好。这是 VGG 系列的核心思想。

**（2）为什么 `Flatten` 后是 2048？**
64×128 的图经过 5 次 2×2 池化后变成 2×4，通道数是 256，
所以 `2 × 4 × 256 = 2048`。

**（3）为什么要 BatchNorm？**
每层卷积输出的数值分布会随层数加深而变化（内部协变量偏移），
BatchNorm 把每一层的输出重新标准化，让训练更快更稳。
它放在卷积后、ReLU 前。

**（4）为什么展平之后直接接分类头，中间没有共享的全连接层？**
这是本项目实测踩过的坑，详见下方第 5 节。
一句话：Flatten 出来的 2048 维里带着「哪块区域对应哪一位」的信息，
中间再压成 512 维会把这部分空间信息抹掉，收敛会慢一个数量级。

**（4）为什么输出不加 softmax？**
交叉熵损失内部自带 `log_softmax`，自己再加一次就重复了。

```python
# 对的
loss = criterion(logits, labels)
# 错的（softmax 被算了两次）
loss = criterion(torch.softmax(logits, dim=1), labels)
```

### 3.4 `train.py` — 训练

**两阶段学习率**（原参考项目里用过的技巧）：

1. **阶段 1**：学习率 `1e-3`，快速收敛
2. **阶段 2**：学习率降到 `1e-4`，精修

每个阶段都有 EarlyStopping（验证集准确率连续 8 轮不提升就停），
始终保存验证准确率最好的那份权重。

**损失函数是 4 个交叉熵之和**：

```python
def compute_loss(outputs, labels, criterion):
    loss = 0.0
    for i, logits in enumerate(outputs):
        loss = loss + criterion(logits, labels[:, i])
    return loss
```

`labels` 形状是 `(B, 4)`，第 i 列是第 i 位字符的标签；
`labels[:, i]` 取出来正好对应第 i 个分类头。

### 3.5 `predict.py` / `evaluate.py`

- `predict.py`：生成若干张验证码并识别，画出对比图（对的绿标题、错的红标题）
- `evaluate.py`：统计整体准确率和每一位的准确率

---

## 4. 两个准确率的区别

这是本项目最值得讲的一个知识点。

- **单字符准确率**：每一位单独统计，比如第 1 位对了 99%
- **整体准确率**：4 位全对才算这张对

假设每一位都有 99% 的准确率，整体准确率是：

\[
0.99^4 \approx 96.1\%
\]

也就是说，**每位 99% 并不意味着整体 99%**。
这也解释了为什么验证码识别比单字符识别难得多。

`evaluate.py` 的真实输出（本项目训练后的模型，3200 张全新样本）：

```
========================================================
评估样本数：3200
整体准确率（4 位全对）：3200/3200 = 100.00%
各位置准确率：
  第 1 位：3200/3200 = 100.00%
  第 2 位：3200/3200 = 100.00%
  第 3 位：3200/3200 = 100.00%
  第 4 位：3200/3200 = 100.00%
========================================================
```

---

## 5. 实测结果

环境：Python 3.13、PyTorch 2.11.0+cu128、NVIDIA RTX 3060 Laptop GPU。

| 轮数 | 学习率 | 训练 loss | 验证 loss | 整体准确率 |
|---|---|---|---|---|
| 1 | 1e-3 | 5.3798 | 0.0199 | 99.61% |
| 2 | 1e-3 | 0.0271 | 0.0025 | 99.97% |
| 4 | 1e-3 | 0.0069 | 0.0002 | **100.00%** |
| 7 | 1e-3 | 0.0030 | 0.0000 | **100.00%** |

模型大小约 10.6 MB，参数量 2,650,416，单轮训练约 55 秒。

参考项目 [ypwhs/captcha_break](https://github.com/ypwhs/captcha_break) 的 Keras 版
报告的整体准确率是 98.26%，本项目略高一些（两边验证码生成器的难度略有不同）。

### 一个很适合讲的对照实验：中间那层共享全连接

最初的版本在 Flatten 之后加了一层 `Linear(2048, 512)` 再接 4 个头。
用同样的 128 张固定样本反复训练，对比收敛速度：

| 训练步数 | 有 512 维瓶颈 | 去掉瓶颈（当前版本） |
|---|---|---|
| 10 | 整体 0.8% | 整体 **61.7%** |
| 20 | 整体 7.8% | 整体 **96.9%** |
| 30 | 整体 35.2% | 整体 **100.0%** |
| 60 | 整体 71.1% | 整体 100.0% |

中间的 512 维瓶颈把空间信息抹掉了，收敛速度差了将近一个数量级。
这个对比很适合让学生体会「网络结构的设计不是随意的」。

---

## 6. 课堂演示建议

### 演示一：看数据长什么样（5 分钟）

```python
from captcha_gen import CaptchaGenerator
import matplotlib.pyplot as plt

gen = CaptchaGenerator()
images, texts = gen.generate_batch(8)

fig, axes = plt.subplots(2, 4, figsize=(10, 3.5))
for ax, img, text in zip(axes.ravel(), images, texts):
    ax.imshow(img.transpose(1, 2, 0))
    ax.set_title(text)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

问学生：为什么要加干扰线和旋转？这是在模拟真实验证码的什么特点？

### 演示二：跟着形状走一遍（10 分钟）

```python
import torch
from model import CaptchaCNN

model = CaptchaCNN()
x = torch.randn(2, 3, 64, 128)

for name, layer in model.features.named_children():
    x = layer(x)
    print(f"{name}: {tuple(x.shape)}")

x = x.flatten(1)
print("flatten:", tuple(x.shape))
x = model.fc(x)
print("fc:", tuple(x.shape))
for i, out in enumerate(model(x)):
    print(f"头{i+1}: {tuple(out.shape)}")
```

### 演示三：对比「每位准确率」和「整体准确率」（5 分钟）

训练完跑 `evaluate.py`，让学生算一下
「如果每位准确率是 99%，整体准确率应该是多少？」。

---

## 7. 常见问题

**Q: 报 `Expected all tensors to be on the same device`？**
模型和输入数据不在同一块设备上。检查 `.to(device)` 有没有漏掉。

**Q: Windows 上 `num_workers > 0` 报错？**
Windows 的多进程用 `spawn` 方式启动，会重新执行整个脚本。
必须把入口包在 `if __name__ == "__main__":` 里，Windows 上建议直接用 `0`。

**Q: 训练很慢？**
数据是现场生成的，生成本身要花时间。可以：
- 用 GPU（`--device cuda`）
- 减少每轮的步数（`--steps-per-epoch`）
- Linux/macOS 上加 `--num-workers 4`

**Q: 只想快速看看流程通不通？**
```bash
python train.py --epochs 2 --steps-per-epoch 20 --no-refine
```

**Q: 准确率一直上不去？**
先确认验证码是不是太难（干扰线太多），可以调 `captcha_gen.py` 里的参数；
再确认 loss 在下降（说明在正常学习，只是数据还不够）。

---

## 8. 参考

- [ypwhs/captcha_break](https://github.com/ypwhs/captcha_break) — 原始 Keras 实现，本项目的思路来源
- PyTorch 官方文档：<https://pytorch.org/docs/stable/index.html>
