# GLM OCR API

A high-performance, production-ready FastAPI microservice designed for advanced Document AI tasks. It combines the reasoning capabilities of **GLM-OCR** (via Ollama) for perfect Markdown extraction with the high-speed geometric detection of **PaddleOCR** and **DocLayout-YOLO**.

## ✨ Features

- **GLM-Powered Markdown OCR**: Directly extract and format complex document layouts, lists, headings, and tables into clean Markdown without artifact generation.
- **High-Speed Text Detection**: Leverages PaddleOCR (PP-OCRv5) for lightning-fast bounding box extraction (skipping the heavy recognition step).
- **Figure & Layout Analysis**: Utilizes DocLayout-YOLO to detect and crop image regions/figures from documents.
- **Production Grade**: Built-in Thread-safe lazy loading for ML models, automatic memory management, smart retry strategies, and comprehensive observability (logging).
- **Easy Configuration**: Highly customizable via `.env` file for CORS, Rate Limits, and Model Prompts.

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ installed and Ollama running locally with the GLM model:
```bash
# Pull the GLM OCR model via Ollama
ollama pull glm-ocr
```

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
python -m venv venv

# Activate on Windows:
.\venv\Scripts\activate
# Activate on Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configuration
Copy the sample environment file and configure it if necessary:
```bash
cp .env.example .env
```
Key configurations inside `.env`:
- `PORT`: Define the API port (default `8000`)
- `CORS_ORIGINS`: Restrict access in production (e.g., `["https://yourfrontend.com"]`)
- `OLLAMA_MODEL`: Target model for text generation (default `glm-ocr`)

### 4. Running the Server

**For Development:**
```bash
python main.py
```

**For Production (Background via nohup):**
```bash
nohup python main.py > api.log 2>&1 &
```
*To monitor logs: `tail -f api.log`*

---

## 📡 API Endpoints

### 1. Full Document OCR (Markdown)
`POST /ocr`
Extracts structural text using the GLM-OCR model.
- **Form Data**:
  - `file`: The image file (JPG, PNG, WebP)
  - `prompt` *(optional)*: Override the default extraction prompt.
- **Response**: Returns pure Markdown text.

### 2. Figure Detection (DocLayout-YOLO)
`POST /paddle-ocr`
Detects images and figures inside the document.
- **Form Data**:
  - `file`: The image file
- **Response**: Returns bounding boxes and confidences for detected figures.

### 3. High-Speed Text Bounding Boxes (PaddleOCR)
`POST /paddle-detect`
Extracts coordinates of all text zones. Skips semantic recognition, making it extremely fast.
- **Form Data**:
  - `file`: The image file
- **Response**: Array of polygons and rectangular bounding boxes.

### 4. System Status
`GET /health`
Returns the status of the FastAPI server, loaded AI models, and Ollama connection health.

---

## 🛠️ Architecture Notes
- **Thread Safety**: YOLO and Paddle models are lazily loaded into memory upon the first request using Python's `threading.Lock()` (Double-checked locking pattern) to prevent GPU/RAM OOM errors during concurrent startups.
- **Image Preprocessing**: All uploaded files are dynamically downscaled and sharpened (via Pillow) before being fed into ML models to optimize latency while preserving visual fidelity.
