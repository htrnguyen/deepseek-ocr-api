import asyncio
import time
from fastapi import APIRouter, UploadFile, File, Form, Request, Depends, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
import ollama as ollama_client

from app.core.config import settings
from app.core.logging import logger
from app.schemas.requests import TranslateRequest
from app.schemas.responses import HealthResponse
from app.services import OCRService, TranslateService, LayoutService, DetectionService

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

ocr_service = OCRService()
translate_service = TranslateService()
layout_service = LayoutService()
detection_service = DetectionService()


async def validate_image(
    file: UploadFile = File(
        ...,
        description="Ảnh đầu vào. Chỉ chấp nhận JPEG, PNG, WebP; kích thước không vượt quá giới hạn cấu hình (MAX_FILE_SIZE).",
    ),
) -> UploadFile:
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, WebP supported")

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        max_mb = settings.MAX_FILE_SIZE // (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"File too large (max {max_mb}MB)")

    file.file_content = content
    return file


_OCR_FORM_PROMPT = Form(
    settings.DEFAULT_PROMPT,
    description="Prompt gửi cho model vision (Ollama). Để mặc định nếu không cần tùy chỉnh.",
)


@router.post(
    "/ocr",
    tags=["OCR — Ollama"],
    summary="OCR ảnh đầy đủ (layout + vision)",
    description=(
        "Pipeline chính: (1) DocLayout YOLO phát hiện vùng ảnh/chữ trên trang; "
        "(2) nếu có vùng ảnh, OCR từng vùng chữ xen kẽ marker `[IMAGE_n]`; "
        "(3) nếu không có vùng ảnh, gọi Ollama một lần trên cả ảnh. "
        "Model OCR lấy từ biến môi trường/cấu hình `OLLAMA_MODEL`."
    ),
)
@limiter.limit(settings.RATE_LIMIT)
async def ocr_endpoint(
    request: Request,
    file: UploadFile = Depends(validate_image),
    prompt: str = _OCR_FORM_PROMPT,
):
    logger.info(f"[Endpoint] POST /ocr file={file.filename}")
    result = await ocr_service.process(file.file_content, file.filename, prompt)
    return result


@router.post(
    "/ocr-layout",
    tags=["OCR — Ollama"],
    summary="OCR ảnh (alias /ocr-layout)",
    description=(
        "**Hiện tại logic giống hệt `POST /ocr`:** cùng gọi `OCRService.process` "
        "(layout DocLayout + OCR Ollama). Giữ endpoint này để tương thích tên cũ hoặc client đã hard-code đường dẫn."
    ),
)
@limiter.limit(settings.RATE_LIMIT)
async def ocr_layout_endpoint(
    request: Request,
    file: UploadFile = Depends(validate_image),
    prompt: str = _OCR_FORM_PROMPT,
):
    logger.info(f"[Endpoint] POST /ocr-layout file={file.filename}")
    result = await ocr_service.process(file.file_content, file.filename, prompt)
    return result


@router.post(
    "/paddle-detect",
    tags=["OCR — Paddle (HTTP)"],
    summary="OCR qua API Paddle bên ngoài",
    description=(
        "Gửi ảnh tới dịch vụ OCR Paddle đặt sẵn trong code (HTTP, `refine=true`). "
        "Không chạy Paddle trực tiếp trong process API; phụ thuộc mạng và máy chủ đích. "
        "Dùng khi bạn muốn kết quả từ pipeline Paddle thay vì Ollama."
    ),
)
@limiter.limit(settings.RATE_LIMIT)
async def detect_endpoint(request: Request, file: UploadFile = Depends(validate_image)):
    logger.info(f"[Endpoint] POST /paddle-detect file={file.filename}")
    result = await detection_service.process(file.file_content, file.filename)
    return result


@router.post(
    "/doclayout",
    tags=["Layout"],
    summary="Phát hiện layout tài liệu (DocLayout YOLO)",
    description=(
        "Chỉ chạy model **DocLayout YOLO** (YOLOv10) trên ảnh: trả về bounding box các vùng "
        "(ví dụ figure, text block…) phục vụ bố cục. **Không** gọi Ollama OCR ở bước này."
    ),
)
@limiter.limit(settings.RATE_LIMIT)
async def layout_endpoint(request: Request, file: UploadFile = Depends(validate_image)):
    logger.info(f"[Endpoint] POST /doclayout file={file.filename}")
    result = await layout_service.process(file.file_content, file.filename)
    return result


@router.post(
    "/paddle-ocr",
    tags=["Layout"],
    summary="Layout (tên legacy /paddle-ocr)",
    description=(
        "**Cùng implementation với `POST /doclayout`:** gọi `LayoutService` (DocLayout YOLO). "
        "Tên endpoint lịch sử; không liên quan tới Paddle inference cục bộ."
    ),
)
@limiter.limit(settings.RATE_LIMIT)
async def layout_legacy_endpoint(
    request: Request, file: UploadFile = Depends(validate_image)
):
    logger.info(f"[Endpoint] POST /paddle-ocr file={file.filename}")
    result = await layout_service.process(file.file_content, file.filename)
    return result


@router.post(
    "/translate",
    tags=["Dịch"],
    summary="Dịch văn bản (Ollama)",
    description=(
        "Dịch chuỗi `text` từ `source_language` sang `target_language` bằng model Ollama "
        "(`OLLAMA_TRANSLATE_MODEL`). Prompt nội bộ giữ nguyên công thức LaTeX ($...$, $$...$$)."
    ),
)
@limiter.limit(settings.RATE_LIMIT)
async def translate_endpoint(request: Request, data: TranslateRequest):
    logger.info(f"[Endpoint] POST /translate text_len={len(data.text)}")
    result = await translate_service.process(
        data.text, data.source_language, data.target_language
    )
    return result


@router.get(
    "/health",
    tags=["Hệ thống"],
    summary="Kiểm tra API và trạng thái Ollama",
    description=(
        "Trả về trạng thái load của các model Ollama cấu hình (`OLLAMA_MODEL`, "
        "`OLLAMA_TRANSLATE_MODEL`): `loaded` / `not_loaded`, hoặc thông báo lỗi nếu không gọi được `ollama ps`."
    ),
)
async def health_endpoint():
    ollama_status = {}
    try:
        ps_response = await asyncio.to_thread(ollama_client.ps)
        running = [m.model for m in getattr(ps_response, "models", [])]
        for name in [settings.OLLAMA_MODEL, settings.OLLAMA_TRANSLATE_MODEL]:
            ollama_status[name] = (
                "loaded" if any(name in n for n in running) else "not_loaded"
            )
    except Exception as e:
        ollama_status = f"error: {type(e).__name__}"

    return HealthResponse(ollama_status=ollama_status)
