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
        clear=True,
    ):
        """初始化生成器的参数。

        characters: 字符集，每个字符占一类
        width/height: 图像宽高（像素）
        n_len: 验证码长度
        font_paths: 字体文件路径列表，依次尝试；None 表示自动查找
        font_size: 字号，None 表示按图像高度自动推算
        clear: True 表示实验用的清晰模式；False 保留较强干扰效果
        """
        self.characters = characters if characters else DEFAULT_CHARACTERS
        self.width = width
        self.height = height
        self.n_len = n_len
        self.font_size = font_size if font_size else max(24, int(height * 0.6))
        self._font_paths = font_paths
        self._font_cache = {}
        self.clear = clear

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
        """画随机干扰线。

        干扰线始终画在文字下方，不覆盖字符笔画。清晰模式只调整线条
        深浅，噪声数量不因此减少。
        """
        count = random.randint(2, 4)
        for _ in range(count):
            x1 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            x2 = random.randint(0, self.width)
            y2 = random.randint(0, self.height)
            if self.clear:
                level = random.randint(150, 215)
                width = random.randint(1, 2)
            else:
                level = random.randint(80, 170)
                width = random.randint(1, 3)
            color = (level, level, level, 255)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

    def _draw_noise(self, image):
        """随机撒一些噪点像素。

        噪点保留原实验的随机数量。清晰模式只避免纯黑或纯白的高对比
        噪点，防止它们看起来像字符笔画；字符本身仍由独立的高分辨率
        轮廓层绘制，不会被这些背景噪点弄模糊。
        """
        n_noise = random.randint(80, 160) if self.clear else random.randint(20, 60)
        pixels = image.load()
        for _ in range(n_noise):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if self.clear:
                # 噪声以浅灰为主，少数稍深一点，避免破坏字符轮廓。
                if random.random() < 0.15:
                    level = random.randint(120, 175)
                else:
                    level = random.randint(175, 235)
                pixels[x, y] = (level, level, level, 255)
            else:
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

        字号是**按格子宽度自适应算出来的**：每个字符分到 width/n 像素宽的
        一格，字号必须小于这格，否则字符会溢出格子、被画布边缘裁掉半边。

        排布方式：
          1. 先按自适应字号把所有字符都渲染出来（还没旋转）
          2. 量出它们加起来的总宽度
          3. 整串水平居中放到画布上，再依次摆

        这样既保证了字符不越界（总宽一定放得下），
        又不会像「按格子中心对齐」那样让字符偏到左边、右侧留一大片空白。
        """
        n = len(text)
        # 每个字符占据的横向宽度
        slot = self.width / max(n, 1)

        # 字号自适应：根据字符实际边界框和旋转后的外接矩形留出安全空间。
        # 不能只按格子宽度估算，否则字体自身的 ascent/descent 或某些字形
        # （例如带下伸部的字母）仍可能被裁掉。
        font = self._fit_font(slot * 0.78, self.height * 0.72)

        # 第一步：先把所有字符渲染出来（带旋转），量出总宽度
        rendered = []
        # 清晰模式下只减少旋转和错位，让字符轮廓保持明确；仍保留一定变化，
        # 避免训练集完全固定。
        angle_limit = 18 if self.clear else 20
        y_jitter = self.height // 16 if self.clear else self.height // 12
        for ch in text:
            char_img = self._render_char(ch, font)
            angle = random.uniform(-angle_limit, angle_limit)
            # 直接旋转原始字形。不要在旋转前裁掉透明边距，也不要先缩放
            # 再旋转，否则细笔画容易被重采样削薄或截断。
            rotated = char_img.rotate(
                angle,
                expand=True,
                resample=Image.BICUBIC if self.clear else Image.BILINEAR,
                fillcolor=(0, 0, 0, 0),
            )
            rendered.append(rotated)

        total_w = sum(img.width for img in rendered)
        total_h = max(img.height for img in rendered) if rendered else 0

        # 字符之间的间隙：只算 n-1 个（字符与字符之间），
        # 不能算 n 个，否则最后一个字符会被推出画布右边。
        # 同时左右各留一点边距（margin），避免字符贴边被裁。
        margin = 4
        avail = max(1, self.width - margin * 2)

        # 保险：如果总宽还是超了画布（比如碰巧全是宽字符 W/M/Q），
        # 就把每个字符等比缩小，直到能放下。
        # 这一步保证**任何字符组合都不会被裁**。
        if total_w > avail:
            scale = avail / total_w
            resized = []
            for img in rendered:
                new_w = max(1, int(img.width * scale))
                new_h = max(1, int(img.height * scale))
                # 用超采样重绘后再缩放，避免直接缩放旋转图导致笔画变细。
                resample = Image.LANCZOS if self.clear else Image.BILINEAR
                resized.append(img.resize((new_w, new_h), resample))
            rendered = resized
            total_w = sum(img.width for img in rendered)
            total_h = max(img.height for img in rendered) if rendered else 0

        gap = max(0, (avail - total_w) // max(n - 1, 1))

        # 摆完所有字符和间隙后的实际总宽，再水平居中
        used_w = total_w + gap * (n - 1)
        start_x = max(margin, (self.width - used_w) // 2)
        # 纵向居中，留出随机错位的空间
        base_y = max(0, (self.height - total_h) // 2)

        # 依次摆上去
        x = start_x
        for i, rotated in enumerate(rendered):
            # 随机纵向错位，但错位后必须仍能完整放在画布中。
            dy = random.randint(-y_jitter, y_jitter)
            px = x
            py = base_y + dy

            # 纵向不许越界：这里既限制起点，也保证字符图完整可见。
            py = max(0, min(py, self.height - rotated.height))
            # 横向也不许越界（万一总宽算多了，硬拉回来）
            px = max(0, min(px, self.width - rotated.width))

            # alpha_composite 会按 alpha 通道把字符图混合到画布上，
            # 透明的部分保留原来的背景。
            canvas.alpha_composite(rotated, dest=(px, py))
            x += rotated.width
            # 只有最后一个字符后面不加间隙
            if i < n - 1:
                x += gap

    def _fit_font(self, max_width, max_height):
        """找一个既不超过 max_width 也不超过 max_height 的字号。

        从原始字号往下试，取第一个满足条件的。
        这样字符就不会因为太大而溢出格子 / 被画布裁掉。

        这里用**平均字符宽度**而不是最宽字符来定字号：
        如果按最宽的 `W` 来卡，所有字符都会被压到最小尺寸，
        结果就是「一个字符占 1/4 画布，但字看起来很小」。
        用平均值定字号，既能保证大多数字符放得下，
        又能让字号保持足够大。
        """
        base_font = self._get_font(self.font_size)
        widths = [self._char_width(ch, base_font) for ch in self.characters[:8]]
        avg_width = sum(widths) / max(len(widths), 1)

        size = self.font_size
        while size > 8:
            font = self._get_font(size)
            # 平均宽度随字号线性缩放
            w = avg_width * (size / self.font_size)
            h = self._char_height(self.characters[0], font)
            if w <= max_width and h <= max_height:
                return font
            size -= 1
        return self._get_font(8)

    def _char_width(self, ch, font):
        """量一个字符在指定字体下的实际宽度（像素）。"""
        tmp = Image.new("RGBA", (self.font_size * 3, self.font_size * 3), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp)
        tmp_draw.text((10, 10), ch, font=font, fill=(0, 0, 0, 255))
        bbox = tmp.getbbox()
        if bbox is None:
            return self.font_size
        return bbox[2] - bbox[0]

    def _char_height(self, ch, font):
        """量一个字符在指定字体下的实际高度（像素）。"""
        tmp = Image.new("RGBA", (self.font_size * 3, self.font_size * 3), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp)
        tmp_draw.text((10, 10), ch, font=font, fill=(0, 0, 0, 255))
        bbox = tmp.getbbox()
        if bbox is None:
            return self.font_size
        return bbox[3] - bbox[1]

    def _render_char(self, ch, font):
        """把单个字符渲染到一张透明背景的小图上。

        返回 RGBA 的 PIL.Image。
        """
        # 先用足够大的画布量一下文字实际占多大。
        # 不在字符自身大小上直接绘制，避免字体某些字形的边界恰好被裁掉。
        measure_size = max(self.font_size * 4, 128)
        tmp = Image.new("RGBA", (measure_size, measure_size), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp)
        tmp_draw.text((measure_size // 4, measure_size // 4), ch, font=font, fill=(0, 0, 0, 255))

        bbox = tmp.getbbox()
        if bbox is None:
            # 空白字符（比如空格），退化成一个小方块
            bbox = (0, 0, self.font_size, self.font_size)

        # 给字形四周增加透明边距。Pillow 的 text() 使用字体自身的
        # ascent/descent 定位，部分字体字形可能贴到临时画布边缘；
        # 保留边距可以避免字符上、下或左、右被截断。
        padding = max(3, self.font_size // 6)
        left = max(0, bbox[0] - padding)
        top = max(0, bbox[1] - padding)
        right = min(tmp.width, bbox[2] + padding)
        bottom = min(tmp.height, bbox[3] + padding)
        bbox = (left, top, right, bottom)

        # 裁掉四周多余的透明区域，但保留安全边距。
        # 重新绘制时使用同一张完整测量画布，避免二次绘制坐标不一致导致
        # 字形被裁掉。
        canvas = Image.new("RGBA", (measure_size, measure_size), (0, 0, 0, 0))
        canvas_draw = ImageDraw.Draw(canvas)
        color = (
            random.randint(0, 80),
            random.randint(0, 80),
            random.randint(0, 80),
        )
        canvas_draw.text((measure_size // 4, measure_size // 4), ch, font=font, fill=color + (255,))
        # 只裁掉测量得到的安全区域，保留边距。
        return canvas.crop(bbox)

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
