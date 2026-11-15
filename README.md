# pytorch-captcha-cnn

用 PyTorch + 卷积神经网络从零实现验证码识别。

数据是「现生成现用」的：每次取样本都现场画一张新的验证码，
所以训练集理论上无限，也不存在过拟合到固定图片的问题。

## 环境

项目使用 [uv](https://docs.astral.sh/uv/) 管理虚拟环境和依赖。

```bash
uv sync                      # 创建 .venv 并安装依赖
uv run python -c "import torch; print(torch.__version__)"
```

如果需要用 PyTorch 自带的 CLI：

```bash
uv run python -m torch.utils.collect_env
```

## 目录结构

```
pytorch-captcha-cnn/
├── captcha_gen.py      # 验证码生成器（纯 Pillow + numpy，不依赖 torch）
├── dataset.py          # 把生成器包装成 PyTorch Dataset / DataLoader
├── model.py            # CNN 模型
├── train.py            # 训练脚本
├── fonts/              # 可选：自带的字体文件
├── samples/            # 生成的样本图（git 忽略）
├── figures/            # 训练曲线等图片（git 忽略）
└── checkpoints/        # 模型权重（git 忽略）
```

## 模块说明

### captcha_gen.py

模仿 `ImageCaptcha` 的思路自己实现验证码绘制：

1. 铺一层浅色背景
2. 画 2~4 条随机颜色的干扰线
3. 随机撒一些噪点
4. 每个字符用随机颜色画上去，带 ±20° 随机旋转和上下错位

默认字符集是 10 个数字 + 26 个大写字母（共 36 类），长度 4，尺寸 128×64。

```python
from captcha_gen import CaptchaGenerator

gen = CaptchaGenerator()
image, text = gen.generate()          # 随机文本
image, text = gen.generate("AB12")    # 指定文本
```

### dataset.py

把生成器包成 `Dataset`，每个样本返回 `(图像张量, 标签索引列表)`：

- 图像张量：`float32`，形状 `(3, 64, 128)`，数值在 `[0, 1]`
- 标签：长度为 4 的整数列表，每位字符对应字符集中的下标

`make_dataloaders()` 返回训练和验证两个 `DataLoader`。
Windows 上 `num_workers` 建议设 0（多进程有坑），Linux/macOS 可设 4~8 提速。
