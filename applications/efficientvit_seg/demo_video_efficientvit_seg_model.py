import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
from eval_efficientvit_seg_model import ADE20KDataset, CityscapesDataset, Resize, ToTensor, get_canvas
from torchvision import transforms
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.models.utils import resize
from efficientvit.seg_model_zoo import create_efficientvit_seg_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Đường dẫn tới file video đầu vào, nhập số 0 nếu muốn dùng Webcam")
    parser.add_argument("--dataset", type=str, default="cityscapes", choices=["cityscapes", "ade20k"])
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--crop_size", type=int, default=1024)
    parser.add_argument("--model", type=str, default="efficientvit-seg-l2-cityscapes")
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="Ket_Qua_Video.mp4")
    parser.add_argument("--no_display", action="store_true", help="Tắt cửa sổ xem trực tiếp")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print("Đang khởi động Não Bộ (EfficientViT)...")
    model = create_efficientvit_seg_model(args.model, weight_url=args.weight_url).cuda()
    model.eval()

    # Hỗ trợ mở bằng Webcam nếu người dùng nhập số 0
    if args.video_path == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video_path)

    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video hoặc webcam {args.video_path}.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.video_path != "0" else 0

    if args.dataset == "cityscapes":
        transform = transforms.Compose(
            [
                Resize((args.crop_size, args.crop_size * 2)),
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        class_colors = CityscapesDataset.class_colors
    elif args.dataset == "ade20k":
        transform = transforms.Compose(
            [
                ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        class_colors = ADE20KDataset.class_colors
    else:
        raise NotImplementedError

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    out = None
    if args.output_path and not args.no_display:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        out = cv2.VideoWriter(args.output_path, fourcc, fps if fps > 0 else 30, (width, height))

    print("=========================================")
    print("      HỆ THỐNG PHÂN TÍCH ADAS ĐANG CHẠY  ")
    print(" Mẹo: Bấm phím 'q' trên cửa sổ video để THOÁT")
    print("=========================================")
    
    cv2.namedWindow("EfficientViT - Live ADAS Vision", cv2.WINDOW_NORMAL)

    with torch.inference_mode():
        with tqdm(total=total_frames if total_frames > 0 else None, desc="Processing") as pbar:
            while cap.isOpened():
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR (của cv2) thành RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                data = image
                
                if args.dataset == "ade20k":
                    h, w = image.shape[:2]
                    if h < w:
                        th = args.crop_size
                        tw = math.ceil(w / h * th / 32) * 32
                    else:
                        tw = args.crop_size
                        th = math.ceil(h / w * tw / 32) * 32
                    if th != h or tw != w:
                        data = cv2.resize(image, dsize=(tw, th), interpolation=cv2.INTER_CUBIC)

                data = transform({"data": data, "label": np.ones_like(data)})["data"]
                data = torch.unsqueeze(data, dim=0).cuda()
                
                # Model dự đoán
                output = model(data)
                
                # Resize mảng màu cho vừa lại với kích thước video gốc
                if output.shape[-2:] != image.shape[:2]:
                    output = resize(output, size=image.shape[:2])
                output = torch.argmax(output, dim=1).cpu().numpy()[0]
                
                # Chồng màu lên ảnh gốc
                canvas = get_canvas(image, output, class_colors)
                canvas_bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
                
                # Tính FPS thực tế đang chạy
                process_time = time.time() - start_time
                live_fps = 1.0 / process_time if process_time > 0 else 0
                
                # Vẽ thông số lên màn hình
                cv2.putText(canvas_bgr, f"FPS: {live_fps:.1f}", (30, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas_bgr, f"Model: EfficientViT-L2", (30, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
                
                if not args.no_display:
                    cv2.imshow("EfficientViT - Live ADAS Vision", canvas_bgr)
                    
                    # Bấm 'q' để thoát, bấm 'space' để dừng
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nĐã hủy bằng phím Q!")
                        break
                
                if out is not None:
                    out.write(canvas_bgr)
                pbar.update(1)

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"\n=======================")
    print(f"XONG! Video phân tích thành công đã được lưu tại: {os.path.abspath(args.output_path)}")

if __name__ == "__main__":
    main()
