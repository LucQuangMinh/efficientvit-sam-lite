import argparse
import math
import os
import sys
import time
import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from efficientvit.models.utils import resize
from efficientvit.seg_model_zoo import create_efficientvit_seg_model
from applications.efficientvit_seg.eval_efficientvit_seg_model import CityscapesDataset, Resize, ToTensor, get_canvas

print("Đang khởi động Hạt Nhân AI (EfficientViT-Seg-L2)...")
# Khởi tạo toàn cục để tránh load model nhiều lần
model = create_efficientvit_seg_model("efficientvit-seg-l2-cityscapes").cuda()
model.eval()

transform = transforms.Compose([
    Resize((1024, 2048)), # Cityscapes default
    ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
class_colors = CityscapesDataset.class_colors

def process_image(img):
    if img is None:
        return None
    image = np.array(img.convert("RGB"))
    
    with torch.inference_mode():
        data = transform({"data": image, "label": np.ones_like(image)})["data"]
        data = torch.unsqueeze(data, dim=0).cuda()
        
        output = model(data)
        if output.shape[-2:] != image.shape[:2]:
            output = resize(output, size=image.shape[:2])
            
        output = torch.argmax(output, dim=1).cpu().numpy()[0]
        canvas = get_canvas(image, output, class_colors)
        
    return canvas

def process_video(video_path, progress=gr.Progress()):
    if video_path is None:
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Không thể đọc định dạng Video này!")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = os.path.join(BASE_DIR, "output_adas_video.mp4")
    # Sử dụng H264 để chạy mượt ngay trên Trình duyệt Web của Gradio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30.0, (width, height))

    with torch.inference_mode():
        for i in progress.tqdm(range(total_frames), desc="Đang phân tích từng Khung Hình (Frames)"):
            ret, frame = cap.read()
            if not ret:
                break
                
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            data = transform({"data": image, "label": np.ones_like(image)})["data"]
            data = torch.unsqueeze(data, dim=0).cuda()
            
            output = model(data)
            if output.shape[-2:] != image.shape[:2]:
                output = resize(output, size=image.shape[:2])
                
            output = torch.argmax(output, dim=1).cpu().numpy()[0]
            canvas = get_canvas(image, output, class_colors)
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            
            out.write(canvas_bgr)
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return output_path

# ================================
# THIẾT KẾ GIAO DIỆN WEB CỰC ĐẸP
# ================================
CSS = """
    .container { max-width: 1200px; margin: auto; }
    h1 { text-align: center; color: #1e3a8a; font-family: 'Space Grotesk', sans-serif; }
"""

with gr.Blocks(title="ADAS Vision by EfficientViT", css=CSS, theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# 🚗 Hệ Thống Trí Tuệ Nhân Tạo Tự Lái (ADAS Vision)")
    gr.Markdown("*Tự động nhận diện Phương tiện, Người đi bộ, Biển báo chỉ bằng Camera thường. Tự hào sức mạnh lõi phân tách ngữ nghĩa (Cityscapes) từ Viện hàn lâm MIT.*")
    
    with gr.Tabs():
        with gr.Tab("🖼️ Chế độ Ảnh (Image)"):
            with gr.Row():
                img_in = gr.Image(type="pil", label="Bức ảnh đường phố (Đầu Vào)")
                img_out = gr.Image(label="Hệ thống quét Radar (Đầu Ra)")
            img_btn = gr.Button("🔍 Phân Tích Khung Cảnh", variant="primary")
            img_btn.click(fn=process_image, inputs=[img_in], outputs=[img_out])
            
        with gr.Tab("🎞️ Chế độ Video (Dashcam)"):
            with gr.Row():
                vid_in = gr.Video(label="File Video quay đường phố (Chuẩn .mp4)")
                vid_out = gr.Video(label="Video Màn Hình Trực Quan Đầu Ra")
            vid_btn = gr.Button("🚀 Kích Hoạt Hệ Thống Quét Chuyển Động", variant="primary")
            vid_btn.click(fn=process_video, inputs=[vid_in], outputs=[vid_out])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
