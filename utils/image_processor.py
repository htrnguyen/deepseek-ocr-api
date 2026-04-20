import tempfile
from io import BytesIO
from PIL import Image, ImageOps
from logger import logger


class ImageProcessor:
    """Handle CPU-bound image operations."""

    MAX_IMAGE_SIZE = 1080
    MIN_IMAGE_SIZE = 150
    MAX_ASPECT_RATIO = 5.0

    @classmethod
    def preprocess_image(
        cls, file_content: bytes
    ) -> tuple[str, tuple[int, int], float]:
        """Image preprocessing (Validate, Resize, Convert, Save). Returns tmp_path, original_size, scale."""
        with Image.open(BytesIO(file_content)) as img:
            img = ImageOps.exif_transpose(img)

            original_size = img.size
            width, height = img.size
            logger.info(
                f"[preprocess] | Image: {width}x{height} | "
                f"Mode: {img.mode} | Size: {len(file_content) / 1024:.1f} KB"
            )

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
            if max(img.size) > cls.MAX_IMAGE_SIZE:
                scale = cls.MAX_IMAGE_SIZE / max(img.size)
                new_size = (int(img.width * scale), int(img.height * scale))
                logger.info(
                    f"[preprocess] | Resizing (LANCZOS) | "
                    f"From: {img.size} | To: {new_size} | Scale: {scale:.3f}"
                )
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp_path = tmp.name
                img.save(tmp_path, format="JPEG", quality=95)

            return tmp_path, original_size, scale
