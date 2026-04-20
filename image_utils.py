from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO


def auto_resize_image(image: Image.Image, max_long_side: int) -> Image.Image:
    """Resize image so longest side <= max_long_side, preserving aspect ratio.

    Uses LANCZOS for downscaling (best quality for reducing size).
    Returns original if already within bounds.
    """
    width, height = image.size
    longest = max(width, height)
    if longest > max_long_side:
        scale = max_long_side / longest
        new_w = int(width * scale)
        new_h = int(height * scale)
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image


def enhance_for_ocr(
    image: Image.Image,
    original_resolution: tuple[int, int] | None = None,
) -> Image.Image:
    """Enhance image for OCR with adaptive processing based on image quality.

    High-res images (>= 2000px): Skip sharpening — already sharp enough.
        Adding sharpness creates artifacts that confuse the vision encoder.
    Low-res images (< 2000px): Apply moderate contrast + sharpness boost.

    Args:
        image: PIL Image to enhance.
        original_resolution: Original (width, height) before any resize.
            If None, uses current image size.
    """
    ref_w, ref_h = original_resolution or image.size
    longest_side = max(ref_w, ref_h)

    # Contrast boost — mild, always helpful
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(1.3)

    if longest_side < 2000:
        # Low-res: needs sharpness help
        sharpness_enhancer = ImageEnhance.Sharpness(image)
        image = sharpness_enhancer.enhance(1.5)
    # else: High-res — skip sharpening to avoid artifacts

    return image


def calculate_save_quality(
    original_size_bytes: int,
    processed_size: tuple[int, int],
) -> int:
    """Calculate optimal JPEG save quality based on image characteristics.

    Goal: Keep saved file manageable without losing OCR-critical detail.
    DeepSeek OCR works best with clean, moderately-compressed images.

    Args:
        original_size_bytes: Original file size in bytes.
        processed_size: (width, height) after resize.
    """
    total_pixels = processed_size[0] * processed_size[1]

    if total_pixels > 1_000_000:  # > 1MP after resize
        return 85
    elif total_pixels > 500_000:  # > 0.5MP
        return 90
    else:
        return 95
