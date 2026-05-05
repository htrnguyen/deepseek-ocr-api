import asyncio
import io
import os
import time
import traceback
from PIL import Image
from logger import logger
from chandra.model.schema import BatchInputItem
from chandra.output import parse_layout, parse_markdown
from chandra.settings import settings as chandra_settings
from config import settings

# macOS MPS Optimizations
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def _select_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class ChandraService:
    def __init__(self):
        self._method = settings.CHANDRA_METHOD.lower()
        self._max_tokens = settings.CHANDRA_MAX_TOKENS

        if self._method == "hf":
            import torch
            self._device = _select_device()
            chandra_settings.TORCH_DEVICE = self._device
            # macOS: SDPA is fastest on MPS
            chandra_settings.TORCH_ATTN = "sdpa" if settings.CHANDRA_SDPA_ATTN else None
            chandra_settings.MAX_OUTPUT_TOKENS = self._max_tokens
            chandra_settings.USE_4BIT_QUANT = settings.CHANDRA_USE_4BIT
            self._hf_model = None
        else:
            self._device = "vllm"
            chandra_settings.VLLM_API_BASE = settings.VLLM_API_BASE
            chandra_settings.VLLM_MODEL_NAME = settings.VLLM_MODEL_NAME
            chandra_settings.MAX_OUTPUT_TOKENS = self._max_tokens

        logger.info(f"[Chandra] Backend={self._method}  device={self._device}  max_tokens={self._max_tokens}")

    def load(self):
        """Lazy load HF model (no-op for vLLM — server is external)."""
        if self._method != "hf":
            logger.info(f"[Chandra] vLLM backend — using server at {settings.VLLM_API_BASE}")
            return
        if self._hf_model is None:
            from chandra.model.hf import load_model
            import torch
            logger.info(f"[Chandra] Loading HF model on device={self._device}")
            t0 = time.time()
            self._hf_model = load_model()

            # macOS: torch.compile for MPS if available (PyTorch 2.0+)
            if self._device == "mps" and hasattr(torch, "compile"):
                try:
                    logger.info("[Chandra] Applying torch.compile for MPS optimization...")
                    self._hf_model = torch.compile(self._hf_model, mode="reduce-overhead", fullgraph=False)
                    logger.info("[Chandra] torch.compile applied successfully")
                except Exception as e:
                    logger.warning(f"[Chandra] torch.compile failed: {e}")

            # MPS warmup - critical for consistent performance
            if self._device == "mps":
                self._mps_warmup()

            logger.info(f"[Chandra] HF model ready  load_time={round(time.time()-t0,1)}s")

    def _mps_warmup(self):
        """Warmup MPS to avoid first-call overhead"""
        import torch
        try:
            dummy_input = torch.zeros(1, 3, 224, 224, device="mps")
            torch.mps.synchronize()
            logger.info("[Chandra] MPS warmup completed")
        except Exception as e:
            logger.warning(f"[Chandra] MPS warmup failed: {e}")

    def _clear_mps_cache(self):
        """Clear MPS memory cache to prevent fragmentation on macOS Unified Memory"""
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass  # Non-critical optimization

    def _run_inference_hf(self, image: Image.Image, max_tokens: int, prompt_type: str) -> list:
        """Blocking HF inference — called via asyncio.to_thread."""
        import torch
        from chandra.model.hf import generate_hf
        batch = [BatchInputItem(image=image, prompt_type=prompt_type)]
        with torch.inference_mode():
            return generate_hf(batch=batch, model=self._hf_model, max_output_tokens=max_tokens)

    def _run_inference_vllm(self, image: Image.Image, max_tokens: int, prompt_type: str) -> list:
        """Blocking vLLM inference — called via asyncio.to_thread."""
        from chandra.model.vllm import generate_vllm
        batch = [BatchInputItem(image=image, prompt_type=prompt_type)]
        return generate_vllm(batch=batch, max_output_tokens=max_tokens)

    async def process(
        self,
        file_content: bytes,
        filename: str,
        prompt_type: str = "ocr_layout",
        max_tokens: int | None = None,
    ):
        if self._method == "hf" and self._hf_model is None:
            await asyncio.to_thread(self.load)

        max_tokens = max_tokens or self._max_tokens
        start = time.time()

        # macOS: Clear MPS cache before heavy inference
        if self._device == "mps":
            await asyncio.to_thread(self._clear_mps_cache)

        try:
            image = Image.open(io.BytesIO(file_content)).convert("RGB")
            logger.info(
                f"[Chandra] START  file={filename}  size={image.width}x{image.height}  "
                f"backend={self._method}  max_tokens={max_tokens}"
            )

            infer_start = time.time()
            if self._method == "vllm":
                results = await asyncio.to_thread(self._run_inference_vllm, image, max_tokens, prompt_type)
            else:
                results = await asyncio.to_thread(self._run_inference_hf, image, max_tokens, prompt_type)
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
