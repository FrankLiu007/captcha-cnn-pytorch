"""用训练好的模型识别验证码，并画出对比图。

用法：

    python predict.py                       # 随机生成 12 张并识别
    python predict.py --num 20              # 生成 20 张
    python predict.py --text "AB12"         # 识别指定文本画出来的验证码
    python predict.py --show                # 弹出窗口显示（默认只存文件）

识别的结果会保存到 figures/predictions.png：
正确的那张用绿色标题，错误的那张用红色标题，方便一眼看出模型错在哪。
"""
import argparse
import os

import numpy as np
import torch

from captcha_gen import DEFAULT_CHARACTERS, CaptchaGenerator
from model import CaptchaCNN

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="用训练好的模型识别验证码")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(ROOT, "checkpoints", "captcha_cnn_best.pt"),
        help="模型权重路径",
    )
    parser.add_argument("--num", type=int, default=12, help="生成多少张验证码")
    parser.add_argument(
        "--text", default=None, help="指定要识别的文本（只生成一张）；None 表示随机"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="推理设备",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "figures", "predictions.png"),
        help="图片保存路径",
    )
    parser.add_argument("--show", action="store_true", help="保存后弹出窗口显示")
    return parser.parse_args()


def pick_device(name):
    """选择推理设备，和 train.py 里的逻辑一致。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("指定了 --device cuda，但没有可用的 CUDA 设备。")
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    """加载模型权重。

    权重文件里存了 n_class / n_len，加载时直接用，不用重新指定。
    """
    if not os.path.exists(checkpoint_path):
        raise SystemExit(
            f"找不到权重文件：{checkpoint_path}\n"
            f"请先运行 python train.py 训练模型。"
        )

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CaptchaCNN(
        n_class=payload.get("n_class", len(DEFAULT_CHARACTERS)),
        n_len=payload.get("n_len", 4),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def predict(model, generator, images, device):
    """对一批图像做预测，返回预测出来的字符串列表。"""
    tensor = torch.from_numpy(images).to(device)
    with torch.no_grad():
        outputs = model(tensor)
    # 每个头取 argmax，拼成 (B, n_len)
    preds = torch.stack([o.argmax(dim=1) for o in outputs], dim=1).cpu()
    return [
        "".join(generator.characters[j] for j in row.tolist()) for row in preds
    ]


def plot_predictions(images, texts, predictions, out_path, show=False):
    """画预测结果对比图。"""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plotting import setup_chinese_font

    setup_chinese_font()

    n = len(images)
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.0))
    axes = np.atleast_1d(axes).ravel()

    for ax, image, text, pred in zip(axes, images, texts, predictions):
        # images 是 (B, 3, H, W)，转回 (H, W, 3) 方便 imshow
        ax.imshow(image.transpose(1, 2, 0))
        correct = text == pred
        ax.set_title(
            f"真实 {text}\n预测 {pred}",
            color="#1e7f4f" if correct else "#c0392b",
            fontsize=10,
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    n_correct = sum(t == p for t, p in zip(texts, predictions))
    fig.suptitle(f"验证码识别结果：{n_correct}/{n} 张完全正确", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return n_correct


def main():
    args = parse_args()

    device = pick_device(args.device)
    model, payload = load_model(args.checkpoint, device)
    print(f"设备：{device}")
    print(f"权重：{args.checkpoint}（epoch {payload.get('epoch', '?')}）")

    generator = CaptchaGenerator(characters=DEFAULT_CHARACTERS, n_len=model.n_len)

    if args.text:
        texts = [args.text[: model.n_len].ljust(model.n_len, "0")]
        images, _ = generator.generate_batch(1)
    else:
        images, texts = generator.generate_batch(args.num)

    predictions = predict(model, generator, images, device)

    print("-" * 46)
    for text, pred in zip(texts, predictions):
        mark = "OK " if text == pred else "ERR"
        print(f"[{mark}] 真实 {text}  预测 {pred}")
    print("-" * 46)

    n_correct = sum(t == p for t, p in zip(texts, predictions))
    print(f"整体准确率：{n_correct}/{len(texts)} = {n_correct / len(texts) * 100:.1f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_predictions(images, texts, predictions, args.out, show=args.show)
    print(f"已保存：{args.out}")


if __name__ == "__main__":
    main()
