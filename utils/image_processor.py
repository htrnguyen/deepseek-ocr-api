import tempfile
from io import BytesIO
from PIL import Image, ImageOps
from logger import logger


class ImageProcessor:
    """Handle CPU-bound image operations."""

    MAX_IMAGE_SIZE = 2048

    @classmethod
    def preprocess_image(cls, file_content: bytes) -> tuple[str, tuple[int, int]]:
        """Image preprocessing (Resize, Convert, Save)."""
        with Image.open(BytesIO(file_content)) as img:
            img = ImageOps.exif_transpose(img)

            original_size = img.size
            logger.info(
                f"[process] Image: {original_size[0]}x{original_size[1]} | "
                f"Mode: {img.mode} | Size: {len(file_content) / 1024:.1f} KB"
            )

            if img.mode != "RGB":
                img = img.convert("RGB")

            if max(img.size) > cls.MAX_IMAGE_SIZE:
                ratio = cls.MAX_IMAGE_SIZE / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                logger.info(
                    f"[process] Resizing from {img.size} to {new_size} (LANCZOS)"
                )
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp_path = tmp.name
                img.save(tmp_path, format="JPEG", quality=95)

            return tmp_path, original_size
