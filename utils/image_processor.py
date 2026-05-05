import tempfile
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter, UnidentifiedImageError
from logger import logger


class ImageProcessor:
    """Handle CPU-bound image operations optimized for Grounding & Speed."""

    MAX_IMAGE_SIZE = 1024
    MIN_IMAGE_SIZE = 150
    MAX_ASPECT_RATIO = 5.0
    RETRY_SIZES = [1024, 768, 512]

    # macOS MPS optimizations: Chandra model works best with specific dimensions
    CHANDRA_OPTIMAL_SIZE = 1024  # Chandra uses 1024x1024 internally
    MPS_MAX_SIZE = 1536  # MPS performs better with slightly larger images

    @classmethod
    def preprocess_image(
        cls,
        file_content: bytes,
        max_size: int | None = None,
        skip_validation: bool = False,
    ) -> tuple[str, tuple[int, int], float]:
        """Image preprocessing (Validate, Resize, Convert, Save). Returns tmp_path, original_size, scale."""
        target = max_size or cls.MAX_IMAGE_SIZE

        try:
            img_opened = Image.open(BytesIO(file_content))
        except UnidentifiedImageError:
            raise ValueError("File content is not a valid or supported image format.")

        with img_opened as img:
            img = ImageOps.exif_transpose(img)

            original_size = img.size
            width, height = img.size
            logger.info(
                f"[Image] Preprocess  {width}x{height}  mode={img.mode}  size={len(file_content)/1024:.0f}KB  target={target}px"
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
                    f"[Image] Resize  {img.size[0]}x{img.size[1]} -> max={target}px  scale={scale:.3f}"
                )

                if max(img.size) > target * 2:
                    img.thumbnail((target * 2, target * 2), Image.Resampling.BILINEAR)

                img.thumbnail((target, target), Image.Resampling.LANCZOS)

                img = img.filter(
                    ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3)
                )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp_path = tmp.name
                img.save(tmp_path, format="JPEG", quality=95)

            return tmp_path, original_size, scale
