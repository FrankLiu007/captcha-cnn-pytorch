"""生成 README 里用的验证码样例图。

用法：

    python make_samples.py                       # 存到 docs/captcha_samples.png
    python make_samples.py --num 12 --out xxx.png

图里每张验证码下面标注它对应的文本，方便读者对照着看
「生成器画出来的字符」和「模型要预测的字符」是不是一致。
"""
import argparse
import os

import numpy as np

from captcha_gen import DEFAULT_CHARACTERS, CaptchaGenerator

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="生成验证码样例图")
    parser.add_argument("--num", type=int, default=12, help="生成多少张样例")
    parser.add_argument("--cols", type=int, default=4, help="每行几张")
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "docs", "captcha_samples.png"),
        help="图片保存路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plotting import setup_chinese_font

    setup_chinese_font()

    generator = CaptchaGenerator(characters=DEFAULT_CHARACTERS, n_len=4)
    pairs = [generator.generate() for _ in range(args.num)]

    cols = args.cols
    rows = int(np.ceil(len(pairs) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.3))
    axes = np.atleast_1d(axes).ravel()

    for ax, (image, text) in zip(axes, pairs):
        ax.imshow(np.asarray(image))
        ax.set_title(text, fontsize=11, color="#1f4e79")
        ax.axis("off")

    # 多出来的空格子关掉
    for ax in axes[len(pairs):]:
        ax.axis("off")

    fig.suptitle("验证码生成器随机样例", fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"已保存：{args.out}")


if __name__ == "__main__":
    main()
