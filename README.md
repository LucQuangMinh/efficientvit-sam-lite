# 🚀 EfficientViT-SAM 2.0 (Interactive Edition)

Một ứng dụng AI tối giản, trực quan, khởi chạy trên môi trường Web Local. Ứng dụng này sử dụng sức mạnh của thuật toán **Segment Anything Model (SAM)** được bứt tốc bởi kiến trúc xương sống **EfficientViT** do viện nghiên cứu MIT-Han-Lab phát triển. 

Thay vì phải chạy lệnh code gõ chữ khô khan, ứng dụng cho phép người dùng **tải ảnh lên giao diện Web**, **click chuột vào một vị trí vật thể bất kỳ (Point-Prompt)**, và mạng Nơ-ron AI sẽ lập tức nhận diện rồi vẽ viền bao quanh (Mask) cắt vật thể đó ra với tốc độ hồi đáp chưa tới 1 giây.

---

## ✨ Điểm Nổi Bật của Phiên Bản Này

- **Siêu Nhẹ & Sạch (Zero Bloatware)**: Đã xóa hoàn toàn các đoạn code huấn luyện (Training), thuật toán đo chiều sâu (Depth-Anything), YOLO, hay GazeTracking dư thừa từ siêu dự án gốc. Codebase bây giờ có thể đọc hiểu trong 5 phút.
- **Hiệu Năng Cực Đỉnh**: Nhờ lõi khối MBConv của EfficientViT, tốc độ tự động gom cụm vật thể nhanh hơn đáng kể so với kiến trúc ViT tiêu chuẩn nặng nề của Meta.
- **Giao Diện Trực Quan**: Sử dụng Gradio Web UI với theme cực sáng và rõ thao tác kéo thả/click chuột.
- **Dễ Setup Nhất Có Thể**: Chỉ với 3 bước cài đặt, bất kì dân ngoại đạo công nghệ nào cũng có thể trải nghiệm công nghệ tách viền tiên tiến.

---

## ⚙️ Cấu Trúc Khung AI

```text
D:\100\efficientvit 2\
├── efficientvit/                 # Bộ khung AI thu nhỏ độc lập.
│   ├── models/                   # Lõi thuật toán mạng Neural PyTorch.
│   ├── sam_model_zoo.py          # Script tải kiến trúc mạng EfficientViT-SAM thông minh.
│   └── __init__.py               
├── app.py                        # Giao diện Web App chính.
├── requirements.txt              # Danh sách thư viện siêu mỏng.
└── README.md                     # Tài liệu hướng dẫn cài đặt.
```

---

## 🛠️ Hướng Dẫn Cài Đặt (Dành cho Local Machine windows)

### Bước 1: Chuẩn bị thư viện
Mở Terminal / PowerShell / CMD tại thư mục gốc `efficientvit 2` chạy dòng lệnh sau (Hãy chắc chắn bạn đã cài Python):
```bash
pip install -r requirements.txt
```

### Bước 2: Tải Mô Hình AI Trí Tuệ Phân Tích (Pre-trained Weights)
Vì thuật toán nhận thức hình ảnh nặng tới cả trăm Megabyte, bạn hãy tải file trọng số có sẵn (bản nhẹ và siêu tốc nhất `l0.pt`) bằng đường link chính chủ dưới đây:

🔗 **Link Tải Direct:** [efficientvit_sam_l0.pt (Từ kho của MIT-Han-Lab)](https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt)

✅ **RẤT QUAN TRỌNG**: Sau tải xong, hãy kéo thả file `efficientvit_sam_l0.pt` đó thẳng vào thư mục gốc của dự án này (nằm cùng phòng với file `app.py`).

### Bước 3: Khởi chạy và Trải nghiệm Tương Lai
Bạn chỉ việc gõ dòng lệnh Thần Chú sau vào Terminal:
```bash
python app.py
```
Ngay lập tức, hệ thống sẽ mở Server nội bộ, cung cấp cho bạn đường link Web `http://127.0.0.1:7860`. Copy hoặc Bấm vào link đó! 
- Upload một hình ảnh con Mèo cưng của bạn.
- Bấm vào trán nó.
- Tận hưởng cảm giác AI khoanh vùng chỉ con Mèo xuất hiện đỏ rực trên màn hình mà không cần phải miết tay vẽ viền thủ công bằng Photoshop.

---
*Dự án được đơn giản hóa từ MIT Han Lab's Repo phục vụ cho việc nghiên cứu Core SAM một cách dễ hiểu nhất.*
