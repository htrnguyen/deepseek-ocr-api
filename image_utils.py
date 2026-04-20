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
    """
    import tempfile
    import os
    import time
    from logger import logger

    img_w, img_h = image.size
    logger.info(
        f"[smart_crop] Starting DocLayout-YOLO on {img_w}x{img_h} image"
    )

    # Save temp file for YOLO prediction
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp_path = tmp.name
        image.save(tmp_path, format="JPEG", quality=95)

    try:
        predict_start = time.time()
        results = doclayout_model.predict(
            tmp_path, imgsz=1024, conf=0.2, verbose=False
        )
        predict_time = round(time.time() - predict_start, 3)
        logger.info(f"[smart_crop] DocLayout predict: {predict_time}s")
    finally:
        os.unlink(tmp_path)

    crop_info = {
        "cropped": False,
        "detected_regions": 0,
        "region_labels": [],
    }

    if not results or len(results) == 0:
        logger.info("[smart_crop] No results from DocLayout-YOLO")
        return image, crop_info

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        logger.info("[smart_crop] No boxes detected")
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

    # Log ALL detected regions (including abandoned ones)
    logger.info(f"[smart_crop] Total boxes detected: {len(boxes)}")
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        box = boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        is_content = cls_id in CONTENT_CLASSES
        status = "✓" if is_content else "✗ (abandon)"

        logger.info(
            f"[smart_crop]   {status} [{label}] conf={conf:.2f} "
            f"bbox=[{x1},{y1},{x2},{y2}] size={x2-x1}x{y2-y1}"
        )

        if is_content:
            content_boxes.append(box)
            labels.append(label)

    if not content_boxes:
        logger.info("[smart_crop] No content regions found (all abandoned)")
        return image, crop_info

    # Compute unified bounding box
    import numpy as np
    all_boxes = np.array(content_boxes)
    x1 = int(all_boxes[:, 0].min())
    y1 = int(all_boxes[:, 1].min())
    x2 = int(all_boxes[:, 2].max())
    y2 = int(all_boxes[:, 3].max())

    content_area = (x2 - x1) * (y2 - y1)
    total_area = img_w * img_h
    content_ratio = content_area / total_area

    crop_info["detected_regions"] = len(content_boxes)
    crop_info["region_labels"] = labels
    crop_info["content_ratio"] = round(content_ratio, 2)

    logger.info(
        f"[smart_crop] Content bbox: [{x1},{y1},{x2},{y2}] = "
        f"{x2-x1}x{y2-y1} | "
        f"Content ratio: {content_ratio:.1%} of {img_w}x{img_h}"
    )

    # Skip crop if content already fills most of image
    if content_ratio >= min_content_ratio:
        crop_info["skip_reason"] = (
            f"Content ratio {content_ratio:.0%} >= {min_content_ratio:.0%}"
        )
        logger.info(
            f"[smart_crop] SKIP crop — content {content_ratio:.0%} >= "
            f"threshold {min_content_ratio:.0%}"
        )
        return image, crop_info

    # Apply padding
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(img_w, x2 + padding)
    y2_pad = min(img_h, y2 + padding)

    cropped = image.crop((x1_pad, y1_pad, x2_pad, y2_pad))

    crop_info["cropped"] = True
    crop_info["crop_box"] = [x1_pad, y1_pad, x2_pad, y2_pad]
    crop_info["original_size"] = f"{img_w}x{img_h}"
    crop_info["cropped_size"] = f"{x2_pad - x1_pad}x{y2_pad - y1_pad}"

    logger.info(
        f"[smart_crop] CROP applied: {img_w}x{img_h} → "
        f"{x2_pad - x1_pad}x{y2_pad - y1_pad} | "
        f"Removed {(1 - content_ratio):.0%} empty space"
    )

    return cropped, crop_info
