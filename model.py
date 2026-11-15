"""验证码识别用的多输出 CNN。

结构借鉴 VGG：一个 block 里放两层 3x3 卷积，后面接一次最大池化；
重复几个 block 把特征图尺寸压小、通道数加大，最后展平接全连接。

和多分类不一样的地方：这里有 4 个分类头。
每一层卷积提取的特征是所有字符共用的（它们确实共享笔画、轮廓这些特征），
但最后「这一位是哪个字符」需要各自独立的判断，所以分成 4 个 Linear。
这样也天然得到两个指标：
  - 单字符准确率：每一位分别算
  - 整体准确率：4 位全对才算这一张验证码答对

参考 [ypwhs/captcha_break](https://github.com/ypwhs/captcha_break) 的 Keras 版本。
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """一个卷积块：两层 3x3 卷积 + BatchNorm + ReLU，然后最大池化。

    「两层卷积 + 一次池化」而不是「一层卷积 + 一次池化」，
    是为了在同样的池化次数下拿到更大的感受野，识别效果更好。

    BatchNorm 的作用：验证码的字符颜色是随机的，同一个字符可能是深灰也可能是黑色。
    BatchNorm 把每个通道的分布归一化，等于告诉网络「别太依赖颜色这个线索」，
    强迫它关注形状和笔画结构，训练会稳定很多。
    """

    def __init__(self, in_channels, out_channels, pool=True):
        """in_channels: 输入通道数；out_channels: 输出通道数。

        pool: 是否在块末尾做 2x2 最大池化。设为 False 可以保留分辨率。
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        """前向：卷积 -> BN -> ReLU，重复两次，再池化。"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class CaptchaCNN(nn.Module):
    """验证码识别网络。

    输入: (B, 3, height, width)，例如 (128, 3, 64, 128)
    输出: 一个长度为 n_len 的列表，每个元素是 (B, n_class) 的 logits
          列表第 i 项对应验证码的第 i 位字符

    注意输出的是 logits，不是概率。交叉熵损失内部自带 log_softmax，
    所以这里不要自己加 softmax，否则等于做了两次。
    """

    def __init__(self, n_class=36, n_len=4, height=64, width=128, dropout=0.5):
        """n_class:  字符类别数（默认 36 = 10 数字 + 26 字母）
        n_len:    验证码长度
        height/width: 输入图像尺寸，用来推算展平后的维度
        dropout:  展平后第一层全连接的 dropout 比例
        """
        super().__init__()
        self.n_class = n_class
        self.n_len = n_len

        # 5 个卷积块，通道数 32 -> 64 -> 128 -> 256 -> 256
        # min(i, 3) 让最后两块都是 256，再往上加收益不大，反而参数太多
        blocks = []
        in_channels = 3
        for i in range(5):
            out_channels = 32 * 2 ** min(i, 3)
            blocks.append(ConvBlock(in_channels, out_channels, pool=True))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)

        # 输入 64x128，经过 5 次 2x2 池化后变成 2x4
        self.flat_features = height // 32 * width // 32 * in_channels
        # 展平后直接接 4 个分类头。
        # 参考项目（Keras 版）就是这样做的：Flatten 之后直接接 4 个 Dense，
        # 中间不再加一层共享的全连接。
        # 原因：Flatten 出来的 2048 维里还带着「哪一块区域对应哪一位」的信息，
        # 如果中间压成 512 维，这部分空间信息就被抹掉了，
        # 网络要重新学「第几位字符长什么样」，收敛会慢很多。
        # 4 个头各自从 2048 维里各取所需，互不干扰。
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.flat_features, n_class) for _ in range(n_len)]
        )

        # 一个轻量的 dropout 放在展平之后，防止过拟合。
        # 放在分类头之前（而不是每个头里），保持 4 个头看到相同的输入。
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """初始化权重。

        卷积层用 He 初始化（配合 ReLU），全连接层用 Xavier。
        PyTorch 默认的初始化其实也不差，但显式写出来更符合教学需要，
        也能让学生看到「初始化」不是一个可有可无的细节。
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """前向传播。

        返回长度为 n_len 的列表，第 i 项形状为 (B, n_class)。
        """
        x = self.features(x)          # (B, 256, 2, 4)
        x = x.flatten(1)               # (B, 2048)
        x = self.dropout(x)            # (B, 2048)
        return [head(x) for head in self.classifiers]


def count_parameters(model):
    """统计可训练参数的数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 快速自测：跑一次前向，看看各层形状对不对
    model = CaptchaCNN()
    print(f"参数量：{count_parameters(model):,}")
    x = torch.randn(2, 3, 64, 128)
    # 逐层看形状怎么变（用 feat 另存一份，免得把 x 冲掉）
    feat = x
    for name, layer in model.features.named_children():
        feat = layer(feat)
        print(f"  {name}: {tuple(feat.shape)}")
    outputs = model(x)
    for i, out in enumerate(outputs):
        print(f"第 {i + 1} 位输出形状：{tuple(out.shape)}")