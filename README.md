# DeepSeek OCR API

Production-ready FastAPI service for DeepSeek-OCR + DocLayout-YOLO.

## Quick Start

1. `docker compose up -d --build`
2. Test OCR: `curl -X POST "http://localhost:8000/ocr" -F "file=@test.jpg"`

Endpoints:

- POST /ocr → DeepSeek OCR (markdown)
- POST /paddle-ocr → Figure detection
- GET /health
