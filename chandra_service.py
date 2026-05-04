import asyncio
import io
import time
import torch
import traceback
from PIL import Image
from logger import logger
from chandra.model.hf import load_model, generate_hf
from chandra.model.schema import BatchInputItem
from chandra.output import parse_layout, parse_markdown
from chandra.settings import settings as chandra_settings


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ChandraService:
    def __init__(self):
        self.model = None
        self._device = _select_device()
        chandra_settings.TORCH_DEVICE = self._device
        chandra_settings.TORCH_ATTN = "sdpa"
        chandra_settings.MAX_OUTPUT_TOKENS = 8192

    def load(self):
        """Lazy load model into Unified Memory (MPS/CUDA/CPU)"""
        if self.model is None:
            logger.info(f"[Chandra] Loading model on device={self._device}")
            t0 = time.time()
            self.model = load_model()
            logger.info(f"[Chandra] Model ready  device={self._device}  load_time={round(time.time()-t0,1)}s")

    def _run_inference(self, image: Image.Image) -> list:
        """Blocking inference — must be called inside asyncio.to_thread."""
        batch = [BatchInputItem(image=image, prompt_type="ocr_layout")]
        with torch.inference_mode():
            return generate_hf(batch=batch, model=self.model)

    async def process(
        self, file_content: bytes, filename: str, prompt_type: str = "ocr_layout"
    ):
        if self.model is None:
            await asyncio.to_thread(self.load)

        start = time.time()
        try:
            image = Image.open(io.BytesIO(file_content)).convert("RGB")
            logger.info(
                f"[Chandra] START  file={filename}  size={image.width}x{image.height}  device={self._device}"
            )

            infer_start = time.time()
            results = await asyncio.to_thread(self._run_inference, image)
            infer_time = round(time.time() - infer_start, 2)

            raw_html = results[0].raw
            token_count = results[0].token_count

            parse_start = time.time()
            markdown_text = parse_markdown(raw_html)
            layout_blocks = parse_layout(raw_html, image)
            parse_time = round(time.time() - parse_start, 3)

            blocks_data = [
                {"type": block.label, "bbox": block.bbox}
                for block in layout_blocks
            ]

            total_time = round(time.time() - start, 2)
            logger.info(
                f"[Chandra] DONE   file={filename}  infer={infer_time}s  parse={parse_time}s  "
                f"total={total_time}s  tokens={token_count}  blocks={len(blocks_data)}"
            )

            return {
                "markdown": markdown_text,
                "raw_html": raw_html,
                "blocks": blocks_data,
                "token_count": token_count,
            }

        except Exception as e:
            elapsed = round(time.time() - start, 2)
            logger.error(f"[Chandra] ERROR  file={filename}  time={elapsed}s\n{traceback.format_exc()}")
            raise ValueError(f"Chandra OCR error: {str(e)}")
