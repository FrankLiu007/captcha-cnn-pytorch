"""用 Pillow 自己生成验证码图像。

原参考项目用 `captcha` 这个第三方库生成验证码。这里自己写一份，原因是：
`captcha` 库比较老，在新版 Python 上容易装不上，而且它的代码学生看不到；
自己实现一遍，学生能看清「验证码到底长什么样」，也知道训练数据从哪来。

生成一张验证码的步骤（模仿 ImageCaptcha 的思路）：

    1. 铺一层浅色背景
    2. 画 2~4 条随机颜色的干扰线
    3. 随机撒一些噪点
    4. 把每个字符用随机颜色画上去，带一点随机旋转和错位

这个模块只依赖 Pillow 和 numpy，不依赖 torch，方便单独拿去做数据增强的实验。
"""
import os
import random
import string

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 默认字符集：10 个数字 + 26 个大写字母，一共 36 类
DEFAULT_CHARACTERS = string.digits + string.ascii_uppercase

# 候选字体。按顺序找，找到第一个能用的就用。
# Windows 上一般有 arial.ttf / arialbd.ttf，Linux 上常见 DejaVuSans.ttf。
FONT_CANDIDATES = [
    "arialbd.ttf",
    "arial.ttf",
    "Arial Bold.ttf",
    "DejaVuSans-Bold.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Bold.ttf",
    "LiberationSans-Regular.ttf",
    "consolab.ttf",
    "consola.ttf",
]


def _font_directories():
    """返回可能存在字体文件的目录。

    先看项目里自带的 `fonts/` 文件夹（用户可以往里放字体），
    再看操作系统的字体目录。
    """
    dirs = []
    here = os.path.dirname(os.path.abspath(__file__))
    dirs.append(os.path.join(here, "fonts"))
    if os.name == "nt":
        dirs.append(os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"))
    else:
        dirs.extend(
            [
                "/usr/share/fonts",
                "/usr/local/share/fonts",
                os.path.expanduser("~/.fonts"),
            ]
        )
    return dirs


def find_font_path(explicit=None):
    """找到一个可用的 TrueType 字体文件路径。

    显式传入 explicit 就用它；否则在常见字体目录里按 FONT_CANDIDATES 找。
    找不到就返回 None，调用方需要退回到 Pillow 的默认位图字体。
    """
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"字体文件不存在：{explicit}")
        return explicit

    for directory in _font_directories():
        for name in FONT_CANDIDATES:
            candidate = os.path.join(directory, name)
            if os.path.exists(candidate):
                return candidate
    return None


class CaptchaGenerator:
    """生成验证码图像和对应的文本标签。

    用法：

        gen = CaptchaGenerator()
        image, text = gen.generate()          # 随机文本
        image, text = gen.generate("AB12")    # 指定文本

    输出的图像是 RGB 的 PIL.Image，尺寸为 (height, width)。
    """

    def __init__(
        self,
        characters=DEFAULT_CHARACTERS,
        width=128,
        height=64,
        n_len=4,
        font_paths=None,
        font_size=None,
    ):
        """初始化生成器的参数。

        characters: 字符集，每个字符占一类
        width/height: 图像宽高（像素）
        n_len: 验证码长度
        font_paths: 字体文件路径列表，依次尝试；None 表示自动查找
        font_size: 字号，None 表示按图像高度自动推算
        """
        self.characters = characters if characters else DEFAULT_CHARACTERS
        self.width = width
        self.height = height
        self.n_len = n_len
        self.font_size = font_size if font_size else max(24, int(height * 0.6))
        self._font_paths = font_paths
        self._font_cache = {}

    # ------------------------------------------------------------------
    # 字体
    # ------------------------------------------------------------------
    def _resolve_font_paths(self):
        """确定要用的字体文件列表（结果缓存起来，不必每次都找）。"""
        if self._font_paths is None:
            path = find_font_path()
            self._font_paths = [path] if path else []
        return self._font_paths

    def _get_font(self, size):
        """拿到指定字号的字体对象。找不到系统字体时退回 Pillow 默认字体。"""
        if size in self._font_cache:
            return self._font_cache[size]

        font = None
        for path in self._resolve_font_paths():
            try:
                font = ImageFont.truetype(path, size)
                break
            except OSError:
                continue

        if font is None:
            # Pillow 内置的默认位图字体。字号不可控，但至少能跑通。
            font = ImageFont.load_default()

        self._font_cache[size] = font
        return font

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    def random_text(self):
        """随机生成一个验证码文本。"""
        return "".join(random.choice(self.characters) for _ in range(self.n_len))

    def generate(self, random_str=None):
        """生成一张验证码。

        random_str 为 None 时随机生成；否则用给定的字符串（长度会被截断到 n_len）。
        返回 (PIL.Image, str)。
        """
        if random_str is None:
            random_str = self.random_text()
        random_str = random_str[: self.n_len]

        # 整个过程都在 RGBA 画布上做，最后转回 RGB
        canvas = Image.new("RGBA", (self.width, self.height), self._random_bg())
        draw = ImageDraw.Draw(canvas)

        # 先画干扰线，压在文字下面
        self._draw_lines(draw)
        # 再画噪点
        self._draw_noise(canvas)
        # 最后画文字，压在最上面
        self._draw_text(canvas, random_str)

        return canvas.convert("RGB"), random_str

    def _random_bg(self):
        """随机浅色背景。

        用接近白色的随机颜色，避免背景颜色本身携带信息。
        """
        base = random.randint(225, 250)
        return (
            max(0, min(255, base + random.randint(-8, 8))),
            max(0, min(255, base + random.randint(-8, 8))),
            max(0, min(255, base + random.randint(-8, 8))),
            255,
        )

    def _draw_lines(self, draw):
        """画 2~4 条随机干扰线。

        线的颜色比背景深一点，但不至于完全盖住文字
        （完全盖住的话模型就真的认不出来了）。
        """
        for _ in range(random.randint(2, 4)):
            x1 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            x2 = random.randint(0, self.width)
            y2 = random.randint(0, self.height)
            color = (
                random.randint(80, 170),
                random.randint(80, 170),
                random.randint(80, 170),
                255,
            )
            draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(1, 3))

    def _draw_noise(self, image):
        """随机撒一些噪点像素。

        噪点数量很少，主要是为了增加一点难度，同时不至于影响可读性。
        """
        n_noise = random.randint(20, 60)
        pixels = image.load()
        for _ in range(n_noise):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pixels[x, y] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                255,
            )

    def _draw_text(self, canvas, text):
        """把每个字符画到画布上。

        每个字符：
          - 随机前景色（深色，和浅背景对比明显）
          - 随机旋转 ±20 度
          - 随机上下错位
        """
        font = self._get_font(self.font_size)
        n = len(text)
        # 每个字符占据的横向宽度
        slot = self.width / max(n, 1)

        for i, ch in enumerate(text):
            # 这块字符区域的中点
            cx = int(slot * (i + 0.5))
            cy = int(self.height / 2)

            char_img = self._render_char(ch, font)
            # 随机旋转
            angle = random.uniform(-20, 20)
            rotated = char_img.rotate(angle, expand=True, resample=Image.BILINEAR)

            # 纵向随机错位
            dy = random.randint(-self.height // 12, self.height // 12)
            px = cx - rotated.width // 2
            py = cy - rotated.height // 2 + dy

            # alpha_composite 会按 alpha 通道把字符图混合到画布上，
            # 透明的部分保留原来的背景。
            canvas.alpha_composite(rotated, dest=(px, py))

    def _render_char(self, ch, font):
        """把单个字符渲染到一张透明背景的小图上。

        返回 RGBA 的 PIL.Image。
        """
        # 先用足够大的画布量一下文字实际占多大
        tmp = Image.new("RGBA", (self.font_size * 3, self.font_size * 3), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp)
        tmp_draw.text((10, 10), ch, font=font, fill=(0, 0, 0, 255))

        bbox = tmp.getbbox()
        if bbox is None:
            # 空白字符（比如空格），退化成一个小方块
            bbox = (0, 0, self.font_size, self.font_size)

        # 裁掉四周的透明区域
        tight = tmp.crop(bbox)

        # 重新画一遍到精确大小的图上，填上随机颜色
        canvas = Image.new("RGBA", tight.size, (0, 0, 0, 0))
        canvas_draw = ImageDraw.Draw(canvas)
        color = (
            random.randint(0, 120),
            random.randint(0, 120),
            random.randint(0, 120),
        )
        canvas_draw.text((0, 0), ch, font=font, fill=color + (255,))
        return canvas

    # ------------------------------------------------------------------
    # 批量
    # ------------------------------------------------------------------
    def generate_batch(self, batch_size, return_array=True):
        """一次生成 batch_size 张验证码。

        返回 (images, texts)：
          images: numpy 数组，形状 (batch_size, 3, height, width)，
                  数值已经归一化到 [0, 1]，符合 PyTorch 的输入格式
          texts:  字符串列表，长度 batch_size

        return_array 为 False 时不转 numpy，直接返回 PIL.Image 列表。
        """
        images = []
        texts = []
        for _ in range(batch_size):
            image, text = self.generate()
            images.append(image)
            texts.append(text)

        if not return_array:
            return images, texts

        array = np.stack(
            [np.asarray(img, dtype=np.float32) for img in images]
        )  # (B, H, W, 3)
        # HWC 转 CHW（PyTorch 的约定），再除以 255 归一化到 [0, 1]
        array = array.transpose(0, 3, 1, 2) / 255.0
        return array, texts
