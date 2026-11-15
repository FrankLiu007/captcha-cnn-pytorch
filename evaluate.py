"""评估训练好的模型在验证码上的准确率。

两个指标：

  - 整体准确率：4 位字符全对才算这张验证码答对（这是真正有意义的指标）
  - 单字符准确率：每一位单独统计（能看出是第几位最容易认错）

用法：

    python evaluate.py                       # 默认跑 20 个 batch
    python evaluate.py --batches 100         # 跑得更久，数字更可信
    python evaluate.py --device cpu          # 强制用 CPU
"""
import argparse
import os

import torch

from captcha_gen import DEFAULT_CHARACTERS, CaptchaGenerator
from dataset import CaptchaDataset
from model import CaptchaCNN
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估验证码识别模型的准确率")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(ROOT, "checkpoints", "captcha_cnn_best.pt"),
        help="模型权重路径",
    )
    parser.add_argument("--batches", type=int, default=20, help="评估多少个 batch")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="评估设备",
    )
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    parser.add_argument(
        "--show-errors",
        type=int,
        default=10,
        help="打印多少个识别错误的样本；0 表示不打印",
    )
    return parser.parse_args()


def pick_device(name):
    """选择设备，和 train.py / predict.py 保持一致。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("指定了 --device cuda，但没有可用的 CUDA 设备。")
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    """加载模型权重。"""
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


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    model, payload = load_model(args.checkpoint, device)
    print(f"设备：{device}")
    print(f"权重：{args.checkpoint}（epoch {payload.get('epoch', '?')}）")

    generator = CaptchaGenerator(characters=DEFAULT_CHARACTERS, n_len=model.n_len)
    dataset = CaptchaDataset(generator, length=args.batches * args.batch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    total = 0
    all_correct = 0
    per_position_correct = [0] * model.n_len
    errors = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.stack([o.argmax(dim=1) for o in outputs], dim=1)

            batch_size = images.size(0)
            total += batch_size

            matches = (preds == labels)
            all_correct += matches.all(dim=1).sum().item()
            for i in range(model.n_len):
                per_position_correct[i] += matches[:, i].sum().item()

            # 收集错误样本
            for row in range(batch_size):
                if not matches[row].all():
                    true_text = "".join(
                        generator.characters[j] for j in labels[row].tolist()
                    )
                    pred_text = "".join(
                        generator.characters[j] for j in preds[row].tolist()
                    )
                    errors.append((true_text, pred_text))

    print("=" * 56)
    print(f"评估样本数：{total}")
    print(f"整体准确率（4 位全对）：{all_correct}/{total} = {all_correct / total * 100:.2f}%")
    print("各位置准确率：")
    for i, c in enumerate(per_position_correct):
        print(f"  第 {i + 1} 位：{c}/{total} = {c / total * 100:.2f}%")
    print("=" * 56)

    if args.show_errors and errors:
        print(f"识别错误的样本（前 {min(args.show_errors, len(errors))} 个）：")
        for true_text, pred_text in errors[: args.show_errors]:
            # 标出哪一位错了
            marks = "".join(
                " " if t == p else "^" for t, p in zip(true_text, pred_text)
            )
            print(f"  真实 {true_text}  预测 {pred_text}   {marks}")


if __name__ == "__main__":
    main()
