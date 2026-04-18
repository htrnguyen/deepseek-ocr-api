from PIL import Image, ImageEnhance
from io import BytesIO


def auto_resize_image(image: Image.Image, max_long_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest > max_long_side:
        scale = max_long_side / longest
        new_w = int(width * scale)
        new_h = int(height * scale)
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image


def enhance_for_ocr(image: Image.Image) -> Image.Image:
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(1.5)
    sharpness_enhancer = ImageEnhance.Sharpness(image)
    image = sharpness_enhancer.enhance(2.0)
    return image
