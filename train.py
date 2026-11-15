"""训练验证码识别 CNN。

这个脚本同时支持 CPU 和 GPU：

    python train.py                    # 自动选择（有显卡就用显卡）
    python train.py --device cpu       # 强制用 CPU
    python train.py --device cuda      # 强制用 GPU
    python train.py --epochs 2 --steps-per-epoch 20   # 快速跑一遍看效果

训练的两个关键点：

1. 数据是「现生成现用」的，DataLoader 每取一个 batch 就现场画一批新验证码，
   所以理论上数据无限，不会过拟合到固定的几张图上。

2. 学习率分两段：先用 1e-3 快速收敛，再降到 1e-4 精修。
   这是原参考项目里用过的技巧，效果比一直用一个学习率好。
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from captcha_gen import DEFAULT_CHARACTERS
from dataset import make_dataloaders
from model import CaptchaCNN, count_parameters

# 项目根目录（本文件所在目录）
ROOT = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="训练验证码识别 CNN（CPU / GPU 均可）"
    )
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=500,
        help="每轮训练用多少个 batch（数据无限生成，这里控制单轮时长）",
    )
    parser.add_argument(
        "--val-batches", type=int, default=40, help="每轮验证用多少个 batch"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="第一阶段学习率")
    parser.add_argument(
        "--lr-min", type=float, default=1e-4, help="第二阶段（精修）学习率"
    )
    parser.add_argument(
        "--refine-epochs",
        type=int,
        default=30,
        help="第二阶段训练多少轮；0 表示不开启精修阶段",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练设备。auto 表示有显卡就用显卡",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="数据生成用的子进程数。Windows 建议 0，Linux/macOS 可设 4~8",
    )
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--n-len", type=int, default=4, help="验证码长度")
    parser.add_argument("--width", type=int, default=128, help="图像宽度")
    parser.add_argument("--height", type=int, default=64, help="图像高度")
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="全连接层 dropout 比例"
    )
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "figures"),
        help="图片输出目录",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(ROOT, "checkpoints", "captcha_cnn_best.pt"),
        help="模型权重保存路径",
    )
    parser.add_argument(
        "--log",
        default=os.path.join(ROOT, "checkpoints", "train_log.csv"),
        help="训练日志（CSV）保存路径",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="从已有的权重继续训练",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="跳过第二阶段精修训练",
    )
    return parser.parse_args()


def set_seed(seed):
    """固定随机种子，让结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name):
    """根据参数选择设备。

    auto: 有 CUDA 就用 CUDA，否则用 CPU
    cuda: 强制 CUDA，如果机器上没有可用显卡就报错提示
    cpu: 强制 CPU
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "指定了 --device cuda，但当前环境没有可用的 CUDA 设备。"
                "请检查 PyTorch 是否安装了 CUDA 版本，或改用 --device cpu。"
            )
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------
# 训练与评估
# ----------------------------------------------------------------------
def compute_loss(outputs, labels, criterion):
    """算总损失。

    outputs: 长度 n_len 的列表，每项是 (B, n_class) 的 logits
    labels:  (B, n_len) 的整数张量，每列是那一位字符的类别下标

    每一位各算一个交叉熵，然后求和。求和而不是求平均，
    是因为每一位的预测难度是一样的，权重要一样。
    """
    loss = 0.0
    for i, logits in enumerate(outputs):
        # labels[:, i] 取的是第 i 位字符的标签
        loss = loss + criterion(logits, labels[:, i])
    return loss


def run_epoch(model, loader, device, criterion, optimizer=None):
    """跑一轮训练或验证。

    optimizer 为 None 时只做验证（不更新参数）。

    返回 (平均损失, 整体准确率, 每位字符准确率列表)。
      - 整体准确率：4 位全对才算这张验证码答对
      - 每位准确率：每一位单独统计
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    n_batches = 0

    # 整体正确的数量，和每一位各自正确的数量
    total_samples = 0
    all_correct = 0
    per_position_correct = [0] * model.n_len

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = compute_loss(outputs, labels, criterion)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        n_batches += 1

        # 把每个头的预测取出来拼成 (B, n_len)
        preds = torch.stack([logits.argmax(dim=1) for logits in outputs], dim=1)
        # 每一位都对
        all_correct += (preds == labels).all(dim=1).sum().item()
        # 每一位分别统计
        for i in range(model.n_len):
            per_position_correct[i] += (preds[:, i] == labels[:, i]).sum().item()
        total_samples += batch_size

    mean_loss = total_loss / max(total_samples, 1)
    overall_acc = all_correct / max(total_samples, 1)
    per_position_acc = [c / max(total_samples, 1) for c in per_position_correct]
    return mean_loss, overall_acc, per_position_acc


# ----------------------------------------------------------------------
# 日志与绘图
# ----------------------------------------------------------------------
def write_log(path, epoch, train_loss, val_loss, overall_acc, per_position_acc, lr):
    """把一轮的训练结果追加到 CSV。

    第一次写会先写表头。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["epoch", "lr", "train_loss", "val_loss", "overall_acc"]
                + [f"acc_pos{i + 1}" for i in range(len(per_position_acc))]
            )
        writer.writerow(
            [
                epoch,
                f"{lr:.6g}",
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{overall_acc:.6f}",
            ]
            + [f"{a:.6f}" for a in per_position_acc]
        )


def plot_history(path, out_path):
    """读 CSV 日志，画训练/验证损失和整体准确率曲线。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plotting import setup_chinese_font

    setup_chinese_font()

    epochs, train_losses, val_losses, accs = [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            accs.append(float(row["overall_acc"]) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_losses, label="train loss", color="#1f4e79")
    axes[0].plot(epochs, val_losses, label="val loss", color="#c0392b")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("损失曲线")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, accs, label="整体准确率", color="#1e7f4f")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("验证集整体准确率（4 位全对）")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_checkpoint(model, path, **extra):
    """保存模型权重。

    除了 state_dict，还存了模型超参数，
    这样加载的时候不需要再猜 n_class / n_len 是多少。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "n_class": model.n_class,
        "n_len": model.n_len,
    }
    payload.update(extra)
    torch.save(payload, path)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def train_stage(
    model,
    train_loader,
    val_loader,
    device,
    criterion,
    optimizer,
    epochs,
    lr,
    start_epoch,
    checkpoint_path,
    log_path,
    tag="",
):
    """跑一个训练阶段。

    阶段之间可以换学习率（比如从 1e-3 降到 1e-4）。
    tag 只是打印时用来区分阶段的名字。
    """
    best_acc = -1.0
    patience = 8  # 验证集准确率连续多少轮没提升就提前结束这一阶段
    wait = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        started = time.time()
        # 每个阶段开始时调整学习率
        for group in optimizer.param_groups:
            group["lr"] = lr

        train_loss, _, _ = run_epoch(
            model, train_loader, device, criterion, optimizer=optimizer
        )
        val_loss, val_acc, per_pos = run_epoch(
            model, val_loader, device, criterion, optimizer=None
        )

        write_log(log_path, epoch + 1, train_loss, val_loss, val_acc, per_pos, lr)

        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            wait = 0
            save_checkpoint(
                model,
                checkpoint_path,
                epoch=epoch + 1,
                val_acc=val_acc,
                characters=DEFAULT_CHARACTERS,
            )
        else:
            wait += 1

        pos_str = " ".join(f"{a * 100:5.1f}%" for a in per_pos)
        marker = " *最佳*" if improved else ""
        print(
            f"[{tag}] epoch {epoch + 1:3d} | lr {lr:.1e} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"整体 {val_acc * 100:6.2f}% | 每位 {pos_str} | "
            f"{time.time() - started:.1f}s{marker}"
        )

        if wait >= patience:
            print(f"[{tag}] 验证集准确率连续 {patience} 轮没有提升，提前结束本阶段。")
            break

    return best_acc


def main():
    args = parse_args()

    if args.epochs < 1:
        raise SystemExit("--epochs 至少为 1")

    set_seed(args.seed)
    device = pick_device(args.device)

    print("=" * 70)
    print("验证码识别 CNN 训练")
    print("=" * 70)
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"显卡：{torch.cuda.get_device_name(0)}")
    print(f"字符集：{DEFAULT_CHARACTERS}（{len(DEFAULT_CHARACTERS)} 类）")
    print(f"验证码长度：{args.n_len}    图像尺寸：{args.height}x{args.width}")

    # 数据
    train_loader, val_loader = make_dataloaders(
        characters=DEFAULT_CHARACTERS,
        n_len=args.n_len,
        batch_size=args.batch_size,
        train_steps=args.steps_per_epoch,
        val_batches=args.val_batches,
        num_workers=args.num_workers,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )

    # 模型
    model = CaptchaCNN(
        n_class=len(DEFAULT_CHARACTERS),
        n_len=args.n_len,
        height=args.height,
        width=args.width,
        dropout=args.dropout,
    ).to(device)
    print(f"参数量：{count_parameters(model):,}")
    print(f"每轮训练样本数：{args.steps_per_epoch * args.batch_size:,}")
    print("-" * 70)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    # 续训：先加载权重
    if args.resume and os.path.exists(args.resume):
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        print(f"已从 {args.resume} 载入权重（epoch {payload.get('epoch', '?')}）")
        print("-" * 70)

    # 第一阶段：大步长快速收敛
    best_acc = train_stage(
        model,
        train_loader,
        val_loader,
        device,
        criterion,
        optimizer,
        epochs=args.epochs,
        lr=args.lr,
        start_epoch=0,
        checkpoint_path=args.checkpoint,
        log_path=args.log,
        tag="阶段1",
    )

    # 第二阶段：小步长精修
    if not args.no_refine and args.refine_epochs > 0:
        print("-" * 70)
        print(
            f"进入第二阶段：学习率降到 {args.lr_min:g}，"
            f"继续精修 {args.refine_epochs} 轮"
        )
        # 重新载入最佳权重再精修，避免被训练末期抖动的参数带偏
        if os.path.exists(args.checkpoint):
            payload = torch.load(
                args.checkpoint, map_location=device, weights_only=False
            )
            model.load_state_dict(payload["state_dict"])
        refine_best = train_stage(
            model,
            train_loader,
            val_loader,
            device,
            criterion,
            optimizer,
            epochs=args.refine_epochs,
            lr=args.lr_min,
            start_epoch=args.epochs,
            checkpoint_path=args.checkpoint,
            log_path=args.log,
            tag="阶段2",
        )
        best_acc = max(best_acc, refine_best)

    # 训练结束：载入最佳权重做最终评估
    if os.path.exists(args.checkpoint):
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        print("-" * 70)
        print(f"已载入最佳权重（epoch {payload.get('epoch', '?')}）")

    val_loss, val_acc, per_pos = run_epoch(
        model, val_loader, device, criterion, optimizer=None
    )
    pos_str = " ".join(f"{a * 100:5.1f}%" for a in per_pos)
    print(f"最终验证：loss {val_loss:.4f} | 整体准确率 {val_acc * 100:.2f}% | 每位 {pos_str}")

    # 画图
    os.makedirs(args.out, exist_ok=True)
    history_path = os.path.join(args.out, "training_curves.png")
    try:
        plot_history(args.log, history_path)
        print(f"已保存训练曲线：{history_path}")
    except Exception as exc:  # pragma: no cover - 画图失败不应该影响训练结果
        print(f"画图失败（不影响训练结果）：{exc}")

    print(f"模型权重：{args.checkpoint}")
    print(f"训练日志：{args.log}")
    print("=" * 70)


if __name__ == "__main__":
    # Windows 上 DataLoader 用多进程时，必须把入口包在 main 里，
    # 否则子进程会重新执行整个脚本，导致无限递归。
    main()
