import os
import cv2
import torch
import numpy as np
import gradio as gr

# Thư viện mạng EfficientViT lõi vừa được rút gọn
from efficientvit.sam_model_zoo import create_efficientvit_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

# ==========================================
# 1. CẤU HÌNH MÔ HÌNH (CONFIGURATION)
# ==========================================
# Khuyến khích dùng model l0_pt gọn nhẹ (có thể đổi sang xl1 để tăng mốc chính xác)
MODEL_NAME = "efficientvit-sam-l0"
WEIGHTS_PATH = "./efficientvit_sam_l0.pt"

print("[INFO] Đang khởi tạo hệ thống AI...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Chế độ phần cứng: {device.upper()}")

# Khởi tạo Predictor toàn cục (Global Predictor)
predictor = None

# Kiểm tra người dùng đã tải weights chưa để ra cảnh báo hợp lý thay vì crash
if not os.path.exists(WEIGHTS_PATH):
    print(f"\n[⚠️ CẢNH BÁO] Không tìm thấy file hệ trọng số tại đường dẫn: {WEIGHTS_PATH}")
    print("Vui lòng đọc kỹ hướng dẫn file README.md để tải mô hình trước khi Test ứng dụng trên nền Web.\n")
else:
    print("[INFO] Đang nạp Mạng Nơ-ron EfficientViT-SAM vào VRAM/RAM...")
    
    # Nạp kiến trúc mạng và gán file trọng số (Weights)
    model = create_efficientvit_sam_model(name=MODEL_NAME, weight_url=WEIGHTS_PATH)
    model = model.to(device).eval()
    
    # Bọc kiến trúc qua class Predictor chuyên dụng của bộ SAM
    predictor = EfficientViTSamPredictor(model)
    print("[INFO] Khởi tạo AI hoàn tất. Hệ thống sẵn sàng phục vụ!")


# ==========================================
# 2. HÀM XỬ LÝ LÕI TÁCH NỀN (CORE INFERENCE)
# ==========================================
def process_click(image: np.ndarray, evt: gr.SelectData) -> np.ndarray:
    """
    Xử lý sự kiện Interactive khi người dùng click chuột lên hình ảnh giao diện.
    Hàm sẽ hứng toạ độ Point (X, Y) do User click và gọi Predictor đưa ra Mask.
    """
    if predictor is None:
        raise gr.Error("Chưa load được weights mô hình. Hãy tải file .pt theo hướng dẫn trong README.md nhé!")

    if image is None:
        return None

    # Lấy toạ độ X, Y do mũi tên chuột truyền từ Web Frontend Gradio xuống Backend
    click_x, click_y = evt.index

    # -> Bước 1: Trích xuất đặc trưng của bức hình (Image Embedding)
    predictor.set_image(image)

    # -> Bước 2: Chuẩn bị tín hiệu Prompts (Điểm Chỉ Định)
    # Nhãn số 1 (Label = 1) ám chỉ điểm click này là Điểm Nền Vật Thể (Foreground)
    point_coords = np.array([[click_x, click_y]])
    point_labels = np.array([1])

    # -> Bước 3: Đưa Point vào AI để phân giải, nhận về các viền bao (Masks)
    masks, iou_predictions, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True, # Trả về 3 lớp kết quả (bao quát - trung bình - chật hẹp) để chọn
    )

    # -> Bước 4: AI tự lọc ra Mask có độ tin cậy tự động (IoU - Intersection over Union) cao nhất
    best_mask_idx = np.argmax(iou_predictions)
    best_mask = masks[best_mask_idx]

    # -> Bước 5: Phủ màu lớp màng (Mask) lên không gian ảnh gốc để Review
    # Tạo màu Translucency: Lớp phủ hệ màu Đỏ (RGB: 255, 50, 50) vào các Pixel được đánh dấu bằng True
    color_mask = np.array([255, 50, 50])
    overlay_image = image.copy()
    
    # Pha trộn màu theo công thức Translucency (Độ trong suốt 50%)
    overlay_image[best_mask] = overlay_image[best_mask] * 0.5 + color_mask * 0.5

    # Phóng lại 1 dấu chấm nhỏ Màu Xanh Lá đánh dấu vị trí người dùng đã từng Click
    cv2.circle(overlay_image, (click_x, click_y), radius=6, color=(0, 255, 0), thickness=-1)
    
    # Trả về kết quả đầu ra
    return overlay_image


# ==========================================
# 3. THIẾT KẾ GIAO DIỆN WEB (GRADIO UI)
# ==========================================
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
).set(button_primary_background_fill="*primary_500")

with gr.Blocks(theme=custom_theme, title="EfficientViT SAM 2.0 Web") as demo:
    
    # Header hiển thị chuẩn mực
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>🚀 Mô Hình EfficientViT-SAM 2.0 (Interactive Edition)</h1>
            <p><strong>Công Cụ Tách Nền Point-Prompt</strong> • Hoạt động siêu tốc bằng sức mạnh cục bộ của MIT-Core AI</p>
        </div>
        """
    )
    
    # Body phân chia các bước tương tác rõ ràng
    gr.Markdown(
        """
        ### 📖 Cách trải nghiệm ngay:
        1. **Thả File:** Kéo bức ảnh bất kì trong máy túy của bạn (JPG, PNG) vào khu vực Ô Tải Ảnh bên trái.
        2. **Tương tác Phép Màu:** Xin hãy dùng chuột trỏ và **nhấp Trái (Click)** thật chuẩn xác vào 1 điểm trên khuôn mặt vật thể / xe hơi / động vật. Mạng Nơ-ron sẽ đọc luồng toạ độ (X, Y) và phóng tia bao phủ trọn vẹn sự vật đó dưới 1 giây.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="📸 ẢNH ĐẦU VÀO (Hãy Click trực tiếp lên ảnh này)", interactive=True)
            
        with gr.Column():
            output_image = gr.Image(label="✨ KẾT QUẢ AI PHÂN TÁCH (Phủ Đỏ Mask)", interactive=False)
            
    # Kết nối sự kiện nhấp chuột (Select Event) từ ô Tải Ảnh đưa xuống Backend Python
    input_image.select(fn=process_click, inputs=[input_image], outputs=[output_image])

    # Footer giới thiệu uy tín
    gr.Markdown(
        """
        ---
        <div style="text-align: center; color: gray; font-size: 14px;">
            Hệ thống nguyên bản được phát triển bởi <a href='https://github.com/mit-han-lab/efficientvit' target='_blank'>MIT-Han-Lab</a>. 
            <i>Dự án đã được rút gọn 100% tài nguyên dư thừa, thiết kế cấu trúc gọn nhẹ thích ứng hoàn hảo cho việc Test trực quan Local.</i>
        </div>
        """
    )

# Lệnh khởi chạy server (Localhost)
if __name__ == "__main__":
    demo.launch(inbrowser=True)
