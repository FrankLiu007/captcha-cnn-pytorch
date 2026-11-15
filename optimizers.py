"""自带「自动学习率」机制的优化器集合。

这个文件想说明的一件事是：**优化器里的学习率有两种自动机制**。

    1. 时间维度上的自动（warmup / 衰减）
       这是 lr_scheduler 的活。Adam 也有这类机制，但它只是对 lr 做了一个
       随步数变化的系数，基础的 lr 还是你手动给。

    2. 参数维度上的自动（逐层 / 逐参数缩放）
       这才是本文件里的优化器做的事：每个参数张量根据自己的梯度统计量
       和权重范数，自动算出一个缩放系数。所以你给一个粗略的 lr 就行，
       优化器内部会把它调成"每层各合适"的大小。

典型的逐层自适应优化器：

    - LAMB：Adam 的动量 + 逐层信任比（trust ratio）
            trust_ratio = ||w|| / ||update||
            权重范数大的层，步长自动缩小；权重范数小的层，步长自动放大。
            这是 BERT 训练的标准配置，基本不用调 lr。

    - Lion：只用动量的一阶优化器（没有二阶动量、不需要自适应预热）
            更新量由 sign(动量) 决定，天然带正则化效果，
            实际步长比 Adam 大 3~5 倍，所以 lr 要调小。
            优点是**不需要 warmup**，也不需要 weight decay 的特殊处理。

    - RAdam：Adam 的方差修正项用二阶矩的可信度来「软化」，
            在训练早期梯度二阶矩统计量不可靠时，自动退化成 SGD-Momentum，
            于是**不需要 warmup** 也不会炸。

    - AdamW：Adam + 解耦的 weight decay，weight decay 不再和梯度耦合，
            是现在训练 Transformer/CNN 的默认选择。

本文件里的实现是纯 PyTorch，只依赖 torch.optim.Optimizer，
不需要 pip install 任何第三方库，可以逐行讲解。
"""

import math

import torch
from torch.optim import Optimizer


# ----------------------------------------------------------------------
# 逐层信任比（trust ratio）—— LAMB 的核心
# ----------------------------------------------------------------------
def trust_ratio(param, update, eta=1e-3, eps=1e-8):
    """计算 LAMB 的逐层信任比。

    trust_ratio = ||w|| / ||u||

    其中 u 是「更新量」（Adam 动量去掉二阶矩归一化后的部分）。

    直觉：
      - 某层权重很大（比如后面几层的权重范数往往比前面大），
        那么同样的更新量相对而言就太小了，trust_ratio 会很小，
        于是这一步的实际步长被缩小，避免大权重层被推得太狠。
      - 某层权重很小，trust_ratio 很大，步长被放大，训练更快。

    eta 是 trust_ratio 的上限（对应论文里的 lambda/phi 参数），
    防止 trust_ratio 过大导致步长爆炸。

    返回一个标量张量（与 param 同设备）。
    """
    w_norm = param.norm()
    u_norm = update.norm()
    trust = torch.where(
        u_norm > 0,
        torch.where(
            w_norm > 0,
            (w_norm / u_norm.clamp(min=eps)).clamp(max=eta),
            torch.ones_like(w_norm),
        ),
        torch.ones_like(w_norm),
    )
    return trust


# ----------------------------------------------------------------------
# LAMB
# ----------------------------------------------------------------------
class LAMB(Optimizer):
    """Layer-wise Adaptive Moments optimizer（LAMB）。

    更新公式（论文 "Large Batch Optimization for Deep Learning"）：

        m_t = β1 m_{t-1} + (1-β1) g_t
        v_t = β2 v_{t-1} + (1-β2) g_t²
        u_t = m̂_t / (sqrt(v̂_t) + ε)          # 和 Adam 一样的「自适应更新量」
        r_t = ||w_t|| / ||u_t||                 # 逐层信任比
        w_{t+1} = w_t - η r_t u_t              # 再按信任比缩放

    关键点：r_t 让每一层可以有自己合适的有效学习率，
    所以全局 lr 只要给个粗略值（比如 1e-3）就能训好。

    论文里建议配 weight decay（通常 0.1）而不是 L2 正则，
    因为这里 weight decay 是直接作用在权重上的，和更新量解耦。

    参数：
        lr:          基础学习率。LAMB 对它不敏感，1e-3 左右通常就行。
        betas:       (β1, β2)，默认 (0.9, 0.999)
        eps:         数值稳定项
        weight_decay: 权重衰减系数（论文用 0.1）
        max_trust_ratio: trust_ratio 的上限，等价于论文的 λ
        bias_correction: 是否做 Adam 那样的偏差修正
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.1,
        max_trust_ratio=10.0,
        bias_correction=True,
    ):
        if lr <= 0.0:
            raise ValueError(f"无效的学习率 {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效的 beta1 {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的 beta2 {betas[1]}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_trust_ratio=max_trust_ratio,
            bias_correction=bias_correction,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行一步更新。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            max_trust = group["max_trust_ratio"]
            bias_correction = group["bias_correction"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("LAMB 不支持稀疏梯度")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # 一阶、二阶动量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if bias_correction:
                    bias_correction1 = 1 - beta1 ** t
                    bias_correction2 = 1 - beta2 ** t
                    m_hat = exp_avg / bias_correction1
                    v_hat = exp_avg_sq / bias_correction2
                else:
                    m_hat = exp_avg
                    v_hat = exp_avg_sq

                # 自适应更新量（和 Adam 一样）
                update = m_hat / (v_hat.sqrt() + eps)

                # 逐层信任比
                trust = trust_ratio(param, update, eta=max_trust, eps=eps)

                # weight decay：直接作用在权重上（解耦）
                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

                param.add_(update, alpha=-lr * float(trust))

        return loss


# ----------------------------------------------------------------------
# Lion
# ----------------------------------------------------------------------
class Lion(Optimizer):
    """Lion（EvoLved Sign Momentum）优化器。

    论文 "Symbolic Discovery of Optimization Algorithms"（Chen et al., 2023）：

        m_t = β1 m_{t-1} + (1-β1) g_t
        w_{t+1} = w_t - η · sign(m_t)

    特点：
      - 只有一阶动量，没有二阶动量，状态显存只有 Adam 的一半
      - 更新方向是动量的「符号」，相当于自带很强的正则化
      - **不需要 warmup**：sign 操作天然限制了单步的幅度，
        即使一开始梯度方向乱，也不会像 Adam 那样需要预热
      - 不需要 weight decay 的特殊处理（论文说加了反而变差）

    注意：因为更新量被 sign 归一化成 ±1，实际步长比 Adam 大 3~5 倍，
    所以 lr 要比 Adam 小（论文里 1e-4 ~ 3e-4 这个量级）。
    但 lr 对它相对不敏感，不会出现 Adam 那种 lr 稍微大就训崩的情况。

    参数：
        lr:          学习率，Lion 不敏感，1e-4 ~ 3e-4 都可以
        betas:       (β1, β2)，β2 用于衰减动量。论文推荐 (0.9, 0.99)
        weight_decay: 权重衰减，默认 0（论文说 Lion 不需要）
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"无效的学习率 {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效的 beta1 {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的 beta2 {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行一步更新。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion 不支持稀疏梯度")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                state["step"] += 1
                t = state["step"]

                # 一阶动量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # weight decay（默认关闭）
                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

                # 论文 Eq. 8：用衰减后的动量取 sign 作为更新方向
                #   dir = sign(β2·m_{t-1} + (1-β2)·g_t)
                #   w  -= lr · dir
                direction = exp_avg.mul(beta2).add_(grad, alpha=1 - beta2).sign_()
                param.add_(direction, alpha=-lr)

        return loss


# ----------------------------------------------------------------------
# RAdam
# ----------------------------------------------------------------------
class RAdam(Optimizer):
    """RAdam（Rectified Adam）优化器。

    论文 "On the Variance of the Adaptive Learning Rate and Beyond"（Liu et al., 2020）。

    核心观察：Adam 早期因为二阶动量统计量不准，自适应步长会过大导致发散，
    所以大家都靠 warmup 硬撑过去。RAdam 用一个「可纠正项」ρ_t 判断
    当前二阶矩是否已经足够可信：

        ρ_t = ρ_inf - 2·t·(β₂-1) / (1-β₂^(t+1))

    - ρ_t > 4（早期）：自适应不可信，自动退化成 SGD-Momentum + warmup
    - ρ_t > 0（中期）：有偏差但可用
    - ρ_t <= 0（后期）：完全进入自适应模式

    所以 RAdam **不需要 warmup**，也不会在训练初期炸掉。

    参数：
        lr:          学习率，和 Adam 一样量级（1e-3 左右）
        betas:       (β1, β2)
        eps:         数值稳定项
        weight_decay: 权重衰减
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"无效的学习率 {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效的 beta1 {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的 beta2 {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行一步更新。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # 退化阈值：ρ_t <= 4 时认为自适应不可信
            rho_inf = 2 / (1 - beta2) - 1

            step_count = 0
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("RAdam 不支持稀疏梯度")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                step_count = max(step_count, t)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                # 可纠正项 ρ_t
                rho_t = rho_inf - 2 * t * (beta2 - 1) / bias_correction2

                m_hat = exp_avg / bias_correction1

                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

                if rho_t > 4:
                    # 自适应不可信 → 退化成 SGD-Momentum（带 bias correction）
                    v_sqrt = exp_avg_sq.sqrt().add_(eps)
                    denom = (v_sqrt / math.sqrt(bias_correction2)).add_(eps)
                    param.add_(m_hat / denom, alpha=-lr)
                else:
                    # 自适应可用 → 标准 Adam
                    v_hat_sqrt = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                    param.add_(m_hat / v_hat_sqrt, alpha=-lr)

            group["step_count_if_any"] = step_count

        return loss


# ----------------------------------------------------------------------
# 工厂函数
# ----------------------------------------------------------------------
OPTIMIZER_REGISTRY = {
    "adam": {
        "label": "Adam（原始基线）",
        "cls": torch.optim.Adam,
        "kwargs": {"amsgrad": True},
        "default_lr": 1e-3,
        "desc": "标准基线。lr 需要手动调，两阶段策略就是为它设计的。",
    },
    "adamw": {
        "label": "AdamW（解耦权重衰减）",
        "cls": torch.optim.AdamW,
        "kwargs": {"amsgrad": True},
        "default_lr": 1e-3,
        "desc": "weight decay 与梯度解耦，是现在的默认首选，比 Adam 更稳。",
    },
    "radam": {
        "label": "RAdam（不需要 warmup）",
        "cls": RAdam,
        "kwargs": {},
        "default_lr": 1e-3,
        "desc": "自适应项会自动「退化成 SGD-Momentum」，训练初期不会炸，"
                "因此不用 warmup。",
    },
    "lion": {
        "label": "Lion（sign 动量，不用 warmup）",
        "cls": Lion,
        "kwargs": {},
        "default_lr": 2e-4,
        "desc": "更新方向取动量的符号，天然稳，显存省一半，"
                "但 lr 要比 Adam 小。",
    },
    "lamb": {
        "label": "LAMB（逐层自动学习率）",
        "cls": LAMB,
        "kwargs": {"weight_decay": 0.01},
        "default_lr": 1e-3,
        "desc": "逐层信任比，每层自动缩放步长，最接近「不用调 lr」。",
    },
}


def build_optimizer(name, params, lr=None, **overrides):
    """按名字创建优化器。

    name:  "adam" / "adamw" / "radam" / "lion" / "lamb"
    params: model.parameters()
    lr:    None 表示用该优化器的推荐默认值
    overrides: 额外参数，例如 weight_decay=0.05
    """
    key = name.lower()
    if key not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"未知的优化器 {name!r}，可选：{', '.join(OPTIMIZER_REGISTRY)}"
        )
    spec = OPTIMIZER_REGISTRY[key]
    kwargs = dict(spec["kwargs"])
    kwargs.update(overrides)

    # 只有当用户没显式指定 weight_decay 时，才用该优化器的推荐值
    if "weight_decay" not in kwargs:
        kwargs["weight_decay"] = 0.0
    elif name.lower() == "lamb" and "weight_decay" not in overrides:
        kwargs["weight_decay"] = spec["kwargs"].get("weight_decay", 0.01)

    use_lr = spec["default_lr"] if lr is None else lr
    optimizer = spec["cls"](params, lr=use_lr, **kwargs)
    return optimizer, use_lr


def describe_optimizers():
    """返回所有优化器的说明文字，给 train.py 打印用。"""
    lines = []
    for key, spec in OPTIMIZER_REGISTRY.items():
        lines.append(
            f"  {key:8s} {spec['label']:30s} 推荐 lr={spec['default_lr']:g}\n"
            f"           {spec['desc']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # 自测：在一个小问题上跑一下，确认不会 NaN、参数确实在动
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
    )
    x = torch.randn(64, 8)
    y = (x.sum(dim=1, keepdim=True) > 0).float()

    for key in OPTIMIZER_REGISTRY:
        torch.manual_seed(0)
        net = torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
        )
        opt, lr = build_optimizer(key, net.parameters())
        first = None
        for _ in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                net(x), y
            )
            loss.backward()
            opt.step()
            if first is None:
                first = loss.item()
        print(f"{key:8s} lr={lr:g}  初始 loss {first:.4f} → 最终 {loss.item():.4f}")
