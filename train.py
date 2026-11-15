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
from optimizers import (
    OPTIMIZER_REGISTRY,
    build_optimizer,
    describe_optimizers,
)

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
    parser.add_argument("--lr", type=float, default=None, help="学习率；不填则用所选优化器的推荐值")
    parser.add_argument(
        "--lr-min", type=float, default=1e-4, help="第二阶段（精修）学习率"
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=list(OPTIMIZER_REGISTRY),
        help="优化器。lamb / lion / radam 都自带自动学习率机制",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="权重衰减；不填则用所选优化器的推荐值",
    )
    parser.add_argument(
        "--scheduler",
        default="plateau",
        choices=["none", "plateau", "cosine"],
        help="学习率调度器。plateau=验证损失停滞时自动减半，"
        "cosine=余弦平滑降温，none=不用（完全手动）",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=3,
        help="ReduceLROnPlateau：验证损失多少轮没改善就减半",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau：学习率每次乘以多少",
    )
    parser.add_argument(
        "--scheduler-min-lr",
        type=float,
        default=1e-6,
        help="ReduceLROnPlateau：学习率下限",
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
        default=4,
        help="数据生成用的子进程数。Windows 建议 4 以下，Linux/macOS 可设 8",
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


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None):
    """跑一轮训练或验证。

    optimizer 为 None 时只做验证（不更新参数）。
    scheduler 为 None 时表示这轮不调度学习率；
    否则每个 batch 训练完就调用一次 scheduler.step()（step 级调度）。

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
            # step 级调度器（比如 CosineAnnealingLR）在每个 batch 后走一步
            if scheduler is not None:
                scheduler.step()

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

    epochs, train_losses, val_losses, accs, lrs = [], [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            accs.append(float(row["overall_acc"]) * 100)
            # lr 列可能不存在（老日志），做个兜底
            lrs.append(float(row.get("lr") or 0))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

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

    # 学习率曲线：自动学习率和手动学习率的差别一眼就能看出来
    if any(lr > 0 for lr in lrs):
        axes[2].plot(epochs, lrs, label="学习率", color="#8e44ad")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("学习率变化")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    else:
        axes[2].axis("off")

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
def build_scheduler(scheduler_name, optimizer, args, total_steps):
    """按名字创建学习率调度器。

    返回 (scheduler, mode)：
      - mode == "step"  → 每训练完一个 batch 调用 scheduler.step()
      - mode == "epoch" → 每训练完一轮调用 scheduler.step(metric)
      - mode == None    → 没有调度器

    三种模式的区别很重要：

      plateau（ReduceLROnPlateau）：
        靠「验证指标」驱动，属于 epoch 级调度。
        验证损失连续若干轮不改善就把 lr 乘以 factor（默认 0.5）。
        这是最稳的自动学习率，对本项目这种「数据无限生成、
        验证集一直有噪声」的场景特别合适。

      cosine（CosineAnnealingLR）：
        靠「步数」驱动，属于 step 级调度（每 batch 调一次）。
        学习率沿着余弦曲线从最大学到最小值，平滑且可预期，
        可以完全替代原来的「两阶段手动降 lr」。

      none：
        完全手动，保持原来每轮硬赋值 lr 的行为。
    """
    if scheduler_name == "none":
        return None, None

    if scheduler_name == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=args.scheduler_factor,
                patience=args.scheduler_patience,
                min_lr=args.scheduler_min_lr,
            ),
            "epoch",
        )

    if scheduler_name == "cosine":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=args.scheduler_min_lr,
            ),
            "step",
        )

    raise ValueError(f"未知的调度器 {scheduler_name!r}")


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
    scheduler=None,
    scheduler_mode=None,
    auto_lr=False,
):
    """跑一个训练阶段。

    阶段之间可以换学习率（比如从 1e-3 降到 1e-4）。
    tag 只是打印时用来区分阶段的名字。

    auto_lr 为 True 时，不再每轮硬赋值 lr，而是交给 scheduler 自动调整。
    """
    best_acc = -1.0
    patience = 8  # 验证集准确率连续多少轮没提升就提前结束这一阶段
    wait = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        started = time.time()
        # auto_lr=False 时才是手动设置学习率；
        # auto_lr=True 时由 scheduler 自己控制，这里不碰。
        if not auto_lr:
            for group in optimizer.param_groups:
                group["lr"] = lr

        # 当前学习率（scheduler 模式下会在 step 之后变化，所以先记录）
        cur_lr = optimizer.param_groups[0]["lr"]

        train_loss, _, _ = run_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer=optimizer,
            scheduler=scheduler if scheduler_mode == "step" else None,
        )
        val_loss, val_acc, per_pos = run_epoch(
            model, val_loader, device, criterion, optimizer=None
        )

        # epoch 级调度器：拿验证损失来驱动
        if scheduler is not None and scheduler_mode == "epoch":
            before = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            after = optimizer.param_groups[0]["lr"]
            if after != before:
                print(f"    ↘ 自动调整学习率 {before:.2e} → {after:.2e}")

        write_log(log_path, epoch + 1, train_loss, val_loss, val_acc, per_pos, cur_lr)

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
            f"[{tag}] epoch {epoch + 1:3d} | lr {cur_lr:.1e} | "
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

    # 优化器：可以换成自带「自动学习率」机制的 LAMB / Lion / RAdam
    overrides = {}
    if args.weight_decay is not None:
        overrides["weight_decay"] = args.weight_decay
    optimizer, used_lr = build_optimizer(
        args.optimizer, model.parameters(), lr=args.lr, **overrides
    )
    opt_spec = OPTIMIZER_REGISTRY[args.optimizer]
    print(f"优化器：{opt_spec['label']}")
    print(f"  学习率 {used_lr:g} | weight_decay "
          f"{optimizer.param_groups[0].get('weight_decay', 0.0):g}")
    print(f"  {opt_spec['desc']}")

    # 学习率调度器：负责「学习率大小随训练进程自动变化」这一层
    # total_steps 用来给余弦调度器算一个完整周期
    # （自动模式下只有一段训练，所以总步数就是 epochs × steps_per_epoch）
    if args.scheduler == "none":
        total_steps = args.steps_per_epoch
    else:
        total_steps = args.epochs * args.steps_per_epoch
    scheduler, scheduler_mode = build_scheduler(
        args.scheduler, optimizer, args, total_steps
    )
    if scheduler is None:
        print("学习率调度：手动（沿用原来的两阶段策略）")
    elif scheduler_mode == "epoch":
        print(
            f"学习率调度：ReduceLROnPlateau（验证损失连续 "
            f"{args.scheduler_patience} 轮不改善就 ×{args.scheduler_factor}）"
        )
    else:
        print("学习率调度：CosineAnnealingLR（余弦平滑降温）")

    # 续训：先加载权重
    if args.resume and os.path.exists(args.resume):
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        print(f"已从 {args.resume} 载入权重（epoch {payload.get('epoch', '?')}）")
    print("-" * 70)

    # scheduler 模式下，学习率完全交给 scheduler，不再手动设值，
    # 因此也不再需要「阶段1 → 阶段2」这套手动降 lr 的流程。
    auto_lr = scheduler is not None

    if auto_lr:
        # 自动模式：一个 scheduler 管到底，EarlyStopping 负责收尾。
        # 此时 --epochs 就是总轮数，--refine-epochs 不再单独生效
        # （因为第二阶段的「手动降 lr」已经没有意义了）。
        best_acc = train_stage(
            model,
            train_loader,
            val_loader,
            device,
            criterion,
            optimizer,
            epochs=args.epochs,
            lr=used_lr,
            start_epoch=0,
            checkpoint_path=args.checkpoint,
            log_path=args.log,
            tag="自动学习率",
            scheduler=scheduler,
            scheduler_mode=scheduler_mode,
            auto_lr=True,
        )
    else:
        # 手动模式：保留原来的两阶段策略
        best_acc = train_stage(
            model,
            train_loader,
            val_loader,
            device,
            criterion,
            optimizer,
            epochs=args.epochs,
            lr=used_lr,
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
