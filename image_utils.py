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


def smart_crop_content_region(
    image: Image.Image,
    doclayout_model,
    padding: int = 30,
    min_content_ratio: float = 0.3,
) -> tuple[Image.Image, dict]:
    """Detect content regions via DocLayout-YOLO and crop to bounding area.

    DocLayout-YOLO DocStructBench class IDs:
        0: title, 1: plain text, 2: abandon, 3: figure,
        4: figure_caption, 5: table, 6: table_caption,
        7: table_footnote, 8: isolate_formula, 9: formula_caption

    Strategy:
    - Detect all content classes (0,1,3,4,5,6,7,8,9) — exclude "abandon"(2)
    - Compute unified bounding box around all detected content
    - Add padding and crop
    - If no content detected or content already fills most of the image,
      return original (no crop needed)

    Args:
        image: PIL Image (RGB).
        doclayout_model: Loaded YOLOv10 model instance.
        padding: Pixels to add around detected content region.
        min_content_ratio: If content area / total area > this, skip crop.

    Returns:
        (cropped_image, crop_info) where crop_info has metadata.
    """
    import tempfile
    import os

    # Save temp file for YOLO prediction
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp_path = tmp.name
        image.save(tmp_path, format="JPEG", quality=95)

    try:
        results = doclayout_model.predict(
            tmp_path, imgsz=1024, conf=0.2, verbose=False
        )
    finally:
        os.unlink(tmp_path)

    crop_info = {
        "cropped": False,
        "detected_regions": 0,
        "region_labels": [],
    }

    if not results or len(results) == 0:
        return image, crop_info

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return image, crop_info

    # Content classes (all except "abandon" = 2)
    CONTENT_CLASSES = {0, 1, 3, 4, 5, 6, 7, 8, 9}
    CLASS_NAMES = {
        0: "title", 1: "plain_text", 2: "abandon", 3: "figure",
        4: "figure_caption", 5: "table", 6: "table_caption",
        7: "table_footnote", 8: "isolate_formula", 9: "formula_caption",
    }

    boxes = result.boxes
    content_boxes = []
    labels = []

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        if cls_id in CONTENT_CLASSES:
            box = boxes.xyxy[i].cpu().numpy()
            content_boxes.append(box)
            labels.append(CLASS_NAMES.get(cls_id, f"class_{cls_id}"))

    if not content_boxes:
        return image, crop_info

    # Compute unified bounding box
    import numpy as np
    all_boxes = np.array(content_boxes)
    x1 = int(all_boxes[:, 0].min())
    y1 = int(all_boxes[:, 1].min())
    x2 = int(all_boxes[:, 2].max())
    y2 = int(all_boxes[:, 3].max())

    img_w, img_h = image.size
    content_area = (x2 - x1) * (y2 - y1)
    total_area = img_w * img_h

    crop_info["detected_regions"] = len(content_boxes)
    crop_info["region_labels"] = labels
    crop_info["content_ratio"] = round(content_area / total_area, 2)

    # Skip crop if content already fills most of image
    if content_area / total_area >= min_content_ratio:
        crop_info["skip_reason"] = (
            f"Content ratio {crop_info['content_ratio']:.0%} >= {min_content_ratio:.0%}"
        )
        return image, crop_info

    # Apply padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img_w, x2 + padding)
    y2 = min(img_h, y2 + padding)

    cropped = image.crop((x1, y1, x2, y2))

    crop_info["cropped"] = True
    crop_info["crop_box"] = [x1, y1, x2, y2]
    crop_info["original_size"] = f"{img_w}x{img_h}"
    crop_info["cropped_size"] = f"{x2 - x1}x{y2 - y1}"

    return cropped, crop_info
