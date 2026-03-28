# EfficientViT-Seg (Bản Rút Gọn cho Cityscapes)

Dự án này là phiên bản tối giản và tối ưu của thư viện gốc `efficientvit` (MIT-Han-Lab), được tinh chỉnh đặc biệt để chạy mượt mà trên môi trường Windows phục vụ bài toán **Semantic Segmentation** (Phân rã ngữ nghĩa).

## Các tính năng chính:
- **Chuẩn hóa Cityscapes**: Mã nguồn đã được sửa lỗi không tương thích nhãn (`_labelIds.png` chứa giá trị 255) trong khi Evaluation.
- **Loại bỏ Rác**: Đã cách ly hoàn toàn SAM, Gaze, L2CS-Net và YOLO-NAS để mô hình nguyên khối tập trung 100% vào thư mục thiết yếu.
- **Sẵn sàng sử dụng**: Có sẵn Giao diện Web App (Gradio) phục vụ xử lý Video và Ảnh trực quan. Hoặc bạn có thể dùng Console tùy thích.

## 🚀 Hướng Dẫn Sử Dụng (Quick Start)

### Lựa Chọn 1: Chạy bằng Giao Diện Web Trực Quan (Gradio)
Hỗ trợ cả phân tách Hình ảnh tĩnh (.jpg, .png) lẫn Video (.mp4), tải trực tiếp từ giao diện lên và xem kết quả xuất ra ngay lập tức trên Trình Duyệt.
```bash
python app.py
```

### Lựa Chọn 2: Đánh giá độ chính xác Hệ thống (Evaluation - mIoU)
Sử dụng trên bộ Test / Val của Cityscapes.
```bash
python applications/efficientvit_seg/eval_efficientvit_seg_model.py --dataset cityscapes --model efficientvit-seg-l2-cityscapes --path "D:\Đường_Dẫn_Thư_Mục_Cityscapes\leftImg8bit\val"
```

### Lựa Chọn 3: Trích xuất Video Bằng Terminal (Dành cho Tự Động Hóa)
```bash
python applications/efficientvit_seg/demo_video_efficientvit_seg_model.py --video_path "C:\Video_Của_Bạn.mp4" --dataset cityscapes --crop_size 1024 --model efficientvit-seg-l2-cityscapes --output_path "C:\Video_Mo_Phong_ADAs.mp4"
```

*Lưu ý: Lần chạy đầu tiên hệ thống sẽ tự động quét và tải Weights tự động từ HuggingFace nên cần có Internet.*
