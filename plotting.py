"""画图时的公共设置：让 matplotlib 能显示中文。

matplotlib 默认字体（DejaVu Sans）没有中文字形，直接用中文标题会显示成一堆方框，
还会刷一堆 UserWarning。解决办法是找一个系统里有的中文字体并设为默认。

这一段单独抽出来，是因为 train.py / predict.py / evaluate.py 都要用。
"""
import matplotlib
import matplotlib.pyplot as plt

# 候选中文字体，按优先级排。前面的找不到就试后面的。
# Windows 上一般有 Microsoft YaHei，Linux 上常见 Noto Sans CJK / WenQuanYi。
CJK_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "Noto Sans SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "SimHei",
    "DengXian",
    "KaiTi",
    "FangSong",
]


def setup_chinese_font():
    """把 matplotlib 的默认字体换成系统里可用的中文字体。

    返回实际使用的字体名；一个都没有找到时返回 None（此时中文会显示成方框，
    但不影响程序运行）。
    """
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for name in CJK_FONT_CANDIDATES:
        if name in available:
            # 依次设置这几个入口，matplotlib 各版本行为不完全一样
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["font.family"] = "sans-serif"
            # 负号在有些字体里会缺字形，统一用 ASCII 的减号
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None
