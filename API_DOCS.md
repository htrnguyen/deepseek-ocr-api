# GLM OCR API - Integration Documentation

Tài liệu này hướng dẫn cách tích hợp và sử dụng các endpoints của dịch vụ OCR. Dịch vụ cung cấp 3 tính năng chính: Bóc tách văn bản (Markdown), Trích xuất hình ảnh/biểu đồ, và Quét tọa độ chữ tốc độ cao.

**Base URL mặc định**: `http://localhost:8000`

---

## 1. Bóc tách Văn bản chuẩn Markdown (GLM-OCR)
Sử dụng AI GLM-OCR để đọc toàn bộ nội dung trong ảnh và chuyển đổi thành định dạng Markdown (giữ nguyên bảng biểu, tiêu đề, danh sách).

- **Endpoint**: `POST /ocr`
- **Content-Type**: `multipart/form-data`

### Request (Input)
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | **Yes** | File ảnh cần xử lý (Hỗ trợ: JPG, PNG, WebP. Tối đa: 10MB) |
| `prompt` | String | No | Lệnh tùy chỉnh cho AI. Nếu để trống sẽ dùng prompt mặc định để xuất Markdown. |

### Ví dụ (cURL)
```bash
curl -X POST "http://localhost:8000/ocr" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.jpg"
```

### Response (Output)
```json
{
  "success": true,
  "filename": "document.jpg",
  "text": "# Tiêu đề chính\n\nĐây là đoạn văn bản đã được AI bóc tách...\n\n- Mục 1\n- Mục 2",
  "bboxes": [],
  "original_size": "1024x1448",
  "processing_time": "3.521s",
  "ollama_time": "3.201s",
  "prompt_tokens": 154,
  "response_tokens": 120,
  "total_tokens": 274,
  "status": "success"
}
```
*Lưu ý: Trường `bboxes` được trả về là mảng rỗng `[]` để đảm bảo tính tương thích ngược với API cũ.*

---

## 2. Nhận diện Hình ảnh & Biểu đồ (DocLayout-YOLO)
Quét tài liệu để tìm ra các khu vực chứa hình ảnh, biểu đồ, sơ đồ (bỏ qua phần chữ). Rất hữu ích khi bạn cần cắt hình ảnh ra từ tài liệu gốc.

- **Endpoint**: `POST /paddle-ocr`
- **Content-Type**: `multipart/form-data`

### Request (Input)
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | **Yes** | File ảnh cần xử lý (Tối đa: 10MB) |

### Ví dụ (cURL)
```bash
curl -X POST "http://localhost:8000/paddle-ocr" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@page_with_images.jpg"
```

### Response (Output)
```json
{
  "success": true,
  "filename": "page_with_images.jpg",
  "image_count": 1,
  "images": [
    {
      "label": "image",
      "bbox": [120, 350, 800, 600],
      "confidence": 0.8921,
      "width": 680,
      "height": 250
    }
  ],
  "processing_time": "0.145s",
  "status": "success"
}
```
*(Tọa độ `bbox` định dạng: `[x_min, y_min, x_max, y_max]` tương ứng trên kích thước gốc của ảnh)*

---

## 3. Nhận diện Tọa độ Chữ Tốc độ cao (PaddleOCR Text-Det)
Chỉ thực hiện việc tìm "Khung chữ" (Bounding Box) trên tài liệu mà không dịch ra nội dung chữ (skips recognition). Endpoint này siêu nhanh, dùng để phân tích bố cục hình học của văn bản.

- **Endpoint**: `POST /paddle-detect`
- **Content-Type**: `multipart/form-data`

### Request (Input)
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | **Yes** | File ảnh cần xử lý (Tối đa: 10MB) |

### Ví dụ (cURL)
```bash
curl -X POST "http://localhost:8000/paddle-detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.jpg"
```

### Response (Output)
```json
{
  "success": true,
  "filename": "document.jpg",
  "box_count": 45,
  "boxes": [
    {
      "poly": [[10, 20], [100, 20], [100, 40], [10, 40]],
      "bbox": [10, 20, 100, 40],
      "score": 0.9852
    }
  ],
  "original_size": "1024x1448",
  "detection_time": "0.082s",
  "processing_time": "0.115s",
  "status": "success"
}
```
- `poly`: Tọa độ 4 góc của khung đa giác (có thể bị chéo nếu ảnh nghiêng).
- `bbox`: Tọa độ hình chữ nhật bao quanh (trục tọa độ thẳng).

---

## 4. Kiểm tra Trạng thái Máy chủ (Health Check)
Endpoint này giúp hệ thống (như Load Balancer hoặc Docker Healthcheck) biết API đang hoạt động và xem model `glm-ocr` đã được load lên GPU hay chưa.

- **Endpoint**: `GET /health`

### Response (Output)
```json
{
  "status": "ok",
  "glm_ocr": "enabled",
  "doclayout_yolo": "enabled",
  "paddle_detect": "enabled",
  "ollama_status": "model_loaded",
  "version": "2.5"
}
```
*(Nếu `ollama_status` là `model_not_loaded`, request `/ocr` đầu tiên có thể mất thêm vài giây để nạp model).*

---

## Bảng Mã Lỗi (Error Codes)
Tất cả các API đều trả về HTTP Status Codes chuẩn:
- **`200 OK`**: Thành công.
- **`400 Bad Request`**: File tải lên không đúng định dạng (không phải ảnh) hoặc vượt quá dung lượng (10MB).
- **`422 Unprocessable Entity`**: Lỗi khi model AI bị kẹt hoặc trả về kết quả rỗng (Ảnh quá mờ).
- **`429 Too Many Requests`**: Vượt quá giới hạn (Rate Limit mặc định: 60 requests/phút/IP).
- **`500 / 504 Error`**: Lỗi hệ thống nội bộ hoặc Request Timeout (nếu Model xử lý quá 120s).
