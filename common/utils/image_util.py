import hashlib
import io
from typing import BinaryIO, Optional, Union
from PIL import Image, ImageDraw, ImageFont
from fastapi import UploadFile


def add_watermark(
    image: Image.Image,
    text: str,
    font_size: int | None = 22,
    angle: int = 30,
    alpha: int = 80,
    gap_x: int = 80,   # 同一行文字间距
    gap_y: int = 80,   # 行间距
):
    """
    满屏斜向水印

    参数
    -------
    text       水印文字
    font_size  字体大小
    angle      倾斜角度
    alpha      透明度 0~255
    gap_x      同一行文字间距
    gap_y      行间距
    """

    image = image.convert("RGBA")
    width, height = image.size

    # 扩大画布，避免旋转后出现空白
    canvas_w = int(width * 2)
    canvas_h = int(height * 2)

    watermark = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)

    if font_size is None:
        font_size = max(30, width // 18)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    text_w, text_h = draw.textbbox((0, 0), text, font=font)[2:]

    step_x = text_w + gap_x
    step_y = text_h + gap_y

    y = 0
    while y < canvas_h:
        x = 0
        while x < canvas_w:
            draw.text(
                (x, y),
                text,
                font=font,
                fill=(255, 255, 255, alpha),
            )
            x += step_x
        y += step_y

    # 旋转
    watermark = watermark.rotate(angle, expand=True)

    # 裁剪中心区域
    rw, rh = watermark.size
    left = (rw - width) // 2
    top = (rh - height) // 2

    watermark = watermark.crop((left, top, left + width, top + height))

    result = Image.alpha_composite(image, watermark)

    return result.convert("RGB")


def process_image(
    img: Image.Image,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fit: str = "contain",
):
    """
    图片处理（类似前端 img object-fit）

    fit:
        contain    保持比例完整显示（可能留白）
        cover      填满并裁切
        fill       拉伸填满
        none       不缩放
        scale-down 如果图片大于目标才缩小
    """

    src_w, src_h = img.size

    # 如果都没有传
    if width is None and height is None:
        return img

    # 自动补齐
    width = width or src_w
    height = height or src_h

    # fill：直接拉伸
    if fit == "fill":
        img = img.resize((width, height), Image.LANCZOS)

    # contain：完整显示
    elif fit == "contain":
        img.thumbnail((width, height), Image.LANCZOS)

        new_img = Image.new("RGB", (width, height), (255, 255, 255))
        paste_x = (width - img.width) // 2
        paste_y = (height - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        img = new_img

    # cover：填满裁切
    elif fit == "cover":
        ratio = max(width / src_w, height / src_h)
        new_size = (int(src_w * ratio), int(src_h * ratio))

        img = img.resize(new_size, Image.LANCZOS)

        left = (img.width - width) // 2
        top = (img.height - height) // 2

        img = img.crop((left, top, left + width, top + height))

    # none：不缩放
    elif fit == "none":
        img = img.crop((0, 0, min(width, src_w), min(height, src_h)))

    # scale-down
    elif fit == "scale-down":
        if src_w > width or src_h > height:
            img.thumbnail((width, height), Image.LANCZOS)

    return img


def compress_image(img: Image.Image, quality: int = 85) -> Image.Image:
    """
    压缩图片并返回 Image 对象
    """

    buffer = io.BytesIO()

    img.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True
    )

    buffer.seek(0)

    return Image.open(buffer)


def calc_file_md5(
    f: Union[str, BinaryIO, UploadFile],
    chunk_size: int = 8192
) -> str:
    """
    计算文件 MD5（32位十六进制）
    
    支持：
    - 文件路径字符串
    - UploadFile (FastAPI)
    - file-like 对象 (可读二进制流)
    """
    md5 = hashlib.md5()

    # 文件路径
    if isinstance(f, str):
        with open(f, "rb") as fp:
            for chunk in iter(lambda: fp.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()

    # FastAPI UploadFile
    if isinstance(f, UploadFile):
        f = f.file  # 真实 file-like 对象

    # file-like object
    pos = None
    try:
        pos = f.tell()
    except Exception:
        pass  # 有些流不支持 tell()

    for chunk in iter(lambda: f.read(chunk_size), b""):
        md5.update(chunk)

    try:
        if pos is not None:
            f.seek(pos)  # 恢复指针
    except Exception:
        pass

    return md5.hexdigest()