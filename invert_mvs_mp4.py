# 把影片上下顛倒

import cv2
import os
from tqdm import tqdm

# 輸入與輸出資料夾路徑
input_folder = "mvs_mp4\\20250407\\original"     # 替換成你的輸入資料夾名稱
output_folder = "mvs_mp4\\20250407\\flipped_videos1"  # 影片輸出資料夾

# 建立輸出資料夾（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍歷資料夾中所有 mp4 檔案
for filename in tqdm(os.listdir(input_folder), desc="處理影片", unit="檔案"):
# for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"flipped_{filename}")
        
        # 打開影片
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"無法開啟影片：{input_path}")
            continue

        # 取得影片資訊
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 編碼器
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # print(f"處理影片：{filename}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            flipped_frame = cv2.flip(frame, -1)  # 上下顛倒：0 表示垂直翻轉，-1:上下+左右翻轉
            out.write(flipped_frame)

        cap.release()
        out.release()
        # print(f"儲存成功：{output_path}")

print("所有影片處理完畢。")
