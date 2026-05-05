import tempfile
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter, UnidentifiedImageError
from app.core.logging import logger


class ImageProcessor:
    MAX_IMAGE_SIZE = 1024
    MIN_IMAGE_SIZE = 150
    MAX_ASPECT_RATIO = 5.0

    @classmethod
    def preprocess(cls, file_content: bytes, max_size: int | None = None, skip_validation: bool = False):
        target = max_size or cls.MAX_IMAGE_SIZE

        try:
            img = Image.open(BytesIO(file_content))
        except UnidentifiedImageError:
            raise ValueError("Invalid image format")

        with img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size
            logger.info(f"[Image] {width}x{height} mode={img.mode} {len(file_content)/1024:.0f}KB target={target}px")

            if not skip_validation:
                if min(width, height) < cls.MIN_IMAGE_SIZE:
                    raise ValueError(f"Image too small: {width}x{height}px")
                if max(width, height) / min(width, height) > cls.MAX_ASPECT_RATIO:
                    raise ValueError(f"Extreme aspect ratio: {width}x{height}")

            if img.mode != "RGB":
                img = img.convert("RGB")

            scale = 1.0
            if max(img.size) > target:
                scale = target / max(img.size)
                logger.info(f"[Image] Resize {img.size[0]}x{img.size[1]} -> {target}px scale={scale:.3f}")

                if max(img.size) > target * 2:
                    img.thumbnail((target * 2, target * 2), Image.Resampling.BILINEAR)

                img.thumbnail((target, target), Image.Resampling.LANCZOS)
                img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp_path = tmp.name
                img.save(tmp_path, format="JPEG", quality=95)

            return tmp_path, (width, height), scale
