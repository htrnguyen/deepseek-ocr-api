from PIL import Image, ImageEnhance
from io import BytesIO
import numpy as np
import cv2
from logger import logger


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


def detect_grid_paper(cv_gray: np.ndarray) -> bool:
    """Detect if an image contains grid/graph paper lines.

    Uses Hough Line Transform to count horizontal and vertical lines.
    Grid paper typically has many evenly-spaced lines in both directions.

    Returns True if grid pattern is detected.
    """
    h, w = cv_gray.shape

    # Edge detection
    edges = cv2.Canny(cv_gray, 50, 150, apertureSize=3)

    # Detect lines with HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=int(min(h, w) * 0.15),
                            maxLineGap=10)

    if lines is None:
        return False

    h_lines = 0
    v_lines = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 10 or angle > 170:  # horizontal
            h_lines += 1
        elif 80 < angle < 100:  # vertical
            v_lines += 1

    is_grid = h_lines >= 8 and v_lines >= 8
    logger.info(
        f"[grid_detect] H-lines: {h_lines} | V-lines: {v_lines} | "
        f"Grid detected: {is_grid}"
    )
    return is_grid


def remove_grid_lines(image: Image.Image) -> Image.Image:
    """Remove grid/graph paper lines from an image using morphological operations.

    Algorithm:
    1. Convert to grayscale
    2. Detect horizontal lines via wide horizontal kernel
    3. Detect vertical lines via tall vertical kernel
    4. Combine into a grid mask
    5. Inpaint (fill) the grid lines with surrounding background color
    6. Return cleaned RGB image

    This preserves handwriting strokes (which are short, curved, thick)
    while removing grid lines (which are long, straight, thin).
    """
    # PIL → OpenCV (numpy array)
    cv_img = np.array(image)
    cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    # Check if this image actually has grid lines
    if not detect_grid_paper(cv_gray):
        logger.info("[remove_grid] No grid pattern detected — skipping")
        return image

    logger.info("[remove_grid] Grid pattern detected — removing lines...")

    # Adaptive threshold to get binary image (ink = black, paper = white)
    binary = cv2.adaptiveThreshold(
        cv_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=10
    )

    h, w = binary.shape

    # --- Detect horizontal lines ---
    # Kernel width = 1/20th of image width (catches long lines, ignores short strokes)
    h_kernel_len = max(w // 20, 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)

    # --- Detect vertical lines ---
    v_kernel_len = max(h // 20, 30)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # Combine horizontal + vertical = grid mask
    grid_mask = cv2.add(h_lines, v_lines)

    # Dilate slightly to ensure we cover the full line width
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid_mask = cv2.dilate(grid_mask, dilate_kernel, iterations=1)

    # Inpaint: fill grid line pixels with surrounding colors
    cleaned = cv2.inpaint(cv_img, grid_mask, inpaintRadius=3,
                          flags=cv2.INPAINT_TELEA)

    # Count how many pixels were cleaned
    cleaned_pixels = np.count_nonzero(grid_mask)
    total_pixels = h * w
    cleaned_pct = cleaned_pixels / total_pixels * 100

    logger.info(
        f"[remove_grid] Removed {cleaned_pixels:,} grid pixels "
        f"({cleaned_pct:.1f}% of image)"
    )

    # OpenCV → PIL
    return Image.fromarray(cleaned)


def enhance_for_ocr(image: Image.Image) -> Image.Image:
    """Enhance image for OCR: contrast boost + sharpness recovery.

    Always applies both enhancements since images are typically
    resized to <= 2048px before this step.
    """
    # Contrast boost — mild, always helpful
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(1.3)

    # Sharpness — recover details lost during downscaling
    sharpness_enhancer = ImageEnhance.Sharpness(image)
    image = sharpness_enhancer.enhance(1.5)

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
        # Lower conf to 0.05: We'd rather false-positive a text region (larger crop)
        # than false-negative and chop off faint handwriting on grid paper!
        results = doclayout_model.predict(
            tmp_path, imgsz=1024, conf=0.05, verbose=False
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
        
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / (img_w * img_h)

        # Ignore massive figures/tables (often false positives on background grid paper)
        if is_content and cls_id in {3, 5} and area_ratio >= 0.45:
            logger.info(
                f"[smart_crop]   ✗ (ignore) [{label}] conf={conf:.2f} "
                f"bbox=[{x1},{y1},{x2},{y2}] size={x2-x1}x{y2-y1} "
                f"(covers {area_ratio:.0%} of image, likely background grid!)"
            )
            continue

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

    # Skip crop if content fills most of image
    if content_ratio >= min_content_ratio:
        crop_info["skip_reason"] = (
            f"Content ratio {content_ratio:.0%} >= {min_content_ratio:.0%} (too large)"
        )
        logger.info(f"[smart_crop] SKIP crop — {crop_info['skip_reason']}")
        return image, crop_info

    # Skip crop if content is suspiciously tiny (likely missed faint handwriting)
    if content_ratio < 0.02:
        crop_info["skip_reason"] = (
            f"Content ratio {content_ratio:.1%} < 2% (dangerously small, likely missed text)"
        )
        logger.info(f"[smart_crop] SKIP crop — {crop_info['skip_reason']}")
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
