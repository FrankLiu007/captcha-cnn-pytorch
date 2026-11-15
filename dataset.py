"""把验证码生成器包装成 PyTorch 的 Dataset。

关键点：验证码不需要事先存成文件，可以「现生成现用」。
训练集和验证集用的都是同一个生成器，只是随机种子不同，
所以两张集几乎不会重复，正好符合「无限数据」的想法。
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from captcha_gen import CaptchaGenerator


class CaptchaDataset(Dataset):
    """验证码数据集。

    每个样本是 (图像张量, 标签索引列表)：
      - 图像张量: float32，形状 (3, height, width)，数值在 [0, 1]
      - 标签索引列表: 长度为 n_len 的整数列表，每位字符在字符集中的下标

    注意标签不是 one-hot，而是「每位字符一个整数」。
    因为我们要给每一位配一个独立的分类头，用交叉熵直接吃整数标签更省事。
    """

    def __init__(
        self,
        generator=None,
        length=100000,
        n_class=None,
        characters=None,
    ):
        """初始化数据集。

        generator: 共享的 CaptchaGenerator；None 表示自己建一个
        length:    数据集长度（训练时用 DataLoader 的 steps_per_epoch 控制实际用量）
        n_class:   类别数；None 表示取生成器字符集的长度
        characters: 字符集；generator 是 None 时用来建生成器
        """
        if generator is None:
            generator = CaptchaGenerator(
                characters=characters,
                n_len=4,
            )
        self.generator = generator
        self.length = length
        self.characters = generator.characters
        self.n_class = n_class if n_class else len(self.characters)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.characters)}

    def __len__(self):
        """数据集长度。"""
        return self.length

    def __getitem__(self, index):
        """取出第 index 个样本。

        index 本身没有意义（每次都是新随机生成的），
        这样做的好处是：训练时能无限次取新样本，相当于数据增强。
        """
        image, text = self.generator.generate()
        array = np.asarray(image, dtype=np.float32) / 255.0
        # HWC -> CHW
        array = array.transpose(2, 0, 1)
        tensor = torch.from_numpy(array)

        labels = [self.char_to_idx[ch] for ch in text]
        return tensor, torch.tensor(labels, dtype=torch.long)


def collate_batch(batch):
    """把一批 (tensor, labels) 拼成训练需要的张量。

    DataLoader 默认的 collate 其实已经能做这件事（stack 成 (B, C, H, W) 和 (B, L)），
    这里自己写一个是为了显式说明形状，也方便以后加数据增强。
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    return images, labels


def make_dataloaders(
    characters=None,
    n_len=4,
    batch_size=128,
    train_steps=1000,
    val_batches=40,
    num_workers=0,
    seed=1,
    width=128,
    height=64,
):
    """构造训练和验证两个 DataLoader。

    返回 (train_loader, val_loader)。

    train_steps:   每个 epoch 用多少个 batch（数据是无限生成的，这里控制单轮时长）
    val_batches:  验证集用多少个 batch
    num_workers:  用几个子进程并行生成数据。Windows 上建议 0（多进程有坑），
                  Linux/macOS 上可以设 4、8 提升数据生成速度。
    """
    train_generator = CaptchaGenerator(
        characters=characters, width=width, height=height, n_len=n_len
    )
    val_generator = CaptchaGenerator(
        characters=characters, width=width, height=height, n_len=n_len
    )

    train_dataset = CaptchaDataset(train_generator, length=train_steps * batch_size)
    val_dataset = CaptchaDataset(val_generator, length=val_batches * batch_size)

    # 训练集一直有新数据，所以 shuffle=True
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_batch,
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_batch,
        generator=torch.Generator().manual_seed(seed + 1),
    )

    return train_loader, val_loader
