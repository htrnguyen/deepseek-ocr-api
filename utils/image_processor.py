import time
import os
import tempfile
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
from logger import logger


class ImageProcessor:
    """Handle CPU-bound image operations optimized for Grounding & Speed."""

    MAX_IMAGE_SIZE = 1024
    MIN_IMAGE_SIZE = 150
    MAX_ASPECT_RATIO = 5.0
    RETRY_SIZES = [1024, 768, 512]

    @classmethod
    def preprocess_image(
        cls,
        file_content: bytes,
        max_size: int | None = None,
        skip_validation: bool = False,
    ) -> tuple[str, tuple[int, int], float]:
        """Image preprocessing (Validate, Resize, Convert, Save). Returns tmp_path, original_size, scale."""
        target = max_size or cls.MAX_IMAGE_SIZE

        with Image.open(BytesIO(file_content)) as img:
            img = ImageOps.exif_transpose(img)

            original_size = img.size
            width, height = img.size
            logger.info(
                f"[preprocess] | Image: {width}x{height} | "
                f"Mode: {img.mode} | Size: {len(file_content) / 1024:.1f} KB | "
                f"Target: {target}px"
            )

            if not skip_validation:
                if min(width, height) < cls.MIN_IMAGE_SIZE:
                    raise ValueError(
                        f"Image too small: {width}x{height}px. "
                        f"Minimum dimension is {cls.MIN_IMAGE_SIZE}px."
                    )

                aspect = max(width, height) / min(width, height)
                if aspect > cls.MAX_ASPECT_RATIO:
                    raise ValueError(
                        f"Extreme aspect ratio: {aspect:.1f}:1 ({width}x{height}). "
                        f"Maximum allowed is {cls.MAX_ASPECT_RATIO}:1."
                    )

            if img.mode != "RGB":
                img = img.convert("RGB")

            scale = 1.0
            if max(img.size) > target:
                scale = target / max(img.size)
                logger.info(
                    f"[preprocess] | Resizing | "
                    f"From: {img.size} | Target Max Side: {target}px | Scale: {scale:.3f}"
                )

                img.thumbnail((target, target), Image.Resampling.LANCZOS)
                img = img.filter(
                    ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
                )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp_path = tmp.name
                img.save(tmp_path, format="JPEG", quality=95)

            debug_dir = os.path.join(os.getcwd(), "debug_images")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(
                debug_dir, f"debug_target_{target}px_{int(time.time())}.jpg"
            )
            img.save(debug_path, format="JPEG", quality=95)
            logger.info(f"[preprocess] | Đã lưu ảnh debug: {debug_path}")

            return tmp_path, original_size, scale
