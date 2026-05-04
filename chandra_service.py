import io
import torch
import traceback
from PIL import Image
from logger import logger
from chandra.model.hf import load_model, generate_hf
from chandra.model.schema import BatchInputItem
from chandra.output import parse_layout, parse_markdown
from chandra.settings import settings


class ChandraService:
    def __init__(self):
        self.model = None
        # Tối ưu hóa cho môi trường Mac (Apple Silicon)
        settings.TORCH_DEVICE = "mps"
        settings.TORCH_ATTN = "sdpa"
        settings.MAX_OUTPUT_TOKENS = 8192

    def load(self):
        """Lazy load model vào Unified Memory (MPS)"""
        if self.model is None:
            logger.info(
                "[ChandraService] 🚀 Đang khởi động model Chandra OCR lên MPS..."
            )
            self.model = load_model()
            logger.info("[ChandraService] ✅ Model Chandra đã load thành công!")

    def unload(self):
        """Giải phóng RAM khi không dùng để nhường chỗ cho Ollama"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            logger.info("[ChandraService] 🛑 Đã giải phóng model Chandra khỏi memory.")

    async def process(
        self, file_content: bytes, filename: str, prompt_type: str = "ocr_layout"
    ):
        if self.model is None:
            self.load()

        try:
            image = Image.open(io.BytesIO(file_content)).convert("RGB")

            batch = [BatchInputItem(image=image, prompt_type=prompt_type)]

            logger.info(f"[ChandraService] Đang chạy inference cho {filename}...")
            with torch.inference_mode():
                results = generate_hf(batch=batch, model=self.model)

            raw_html = results[0].raw
            markdown_text = parse_markdown(raw_html)
            layout_blocks = parse_layout(raw_html, image)

            # Chuyển đổi Bbox về dạng JSON
            blocks_data = []
            for block in layout_blocks:
                blocks_data.append(
                    {
                        "type": block.block_type,
                        "bbox": block.bbox,
                    }
                )

            return {
                "markdown": markdown_text,
                "raw_html": raw_html,
                "blocks": blocks_data,
                "token_count": results[0].token_count,
            }

        except Exception as e:
            logger.error(f"[ChandraService] Lỗi xử lý ảnh: {traceback.format_exc()}")
            raise ValueError(f"Chandra OCR error: {str(e)}")
