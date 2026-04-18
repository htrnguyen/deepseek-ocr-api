# DeepSeek OCR API

Production-ready FastAPI service for DeepSeek-OCR + DocLayout-YOLO.

## Quick Start (Native)

1. Tạo virtual environment và cài đặt:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Chạy ngầm (tắt terminal vẫn chạy) và lưu log:
   ```bash
   nohup python main.py > api.log 2>&1 &
   ```

3. Xem log trực tiếp:
   ```bash
   tail -f api.log
   ```

4. Tắt server (Stop):
   ```bash
   lsof -i :8000
   kill -9 <PID>
   ```

Endpoints:
- POST /ocr → DeepSeek OCR (markdown)
- POST /paddle-ocr → Figure detection
- GET /health
