import cv2
import time
from datetime import datetime
import numpy as np
import yaml
import os
from tqdm import tqdm
import subprocess

class VideoRecorder:
    def __init__(self):
        self.camera_index = 1
        self.record_duration = 50 # 錄影秒數
        self.filepath = "crowd_mp4"

        # 讀MV內參
        with open("mvs.yaml", "r") as file:
            mvs_data = yaml.load(file, Loader=yaml.FullLoader)
            self.cameraMatrix = np.array(mvs_data['camera_matrix']['data'])
            self.cameraMatrix = self.cameraMatrix.reshape(3,3)
            self.distortion_coefficients = np.array(mvs_data['distortion_coefficients']['data'])
            self.distortion_coefficients = self.distortion_coefficients.reshape(1,5)
            

    def record(self):
        # 初始化攝影機 (你使用的是 index 1 並指定 DirectShow backend)
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        # 檢查資料夾是否存在
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
            print(f"已建立資料夾：{self.filepath}")

        # 確保攝影機成功開啟
        if not cap.isOpened():
            print("無法開啟攝影機")
            exit()

        # 初始化錄影器
        out = None
        # ffmpeg = None
        recording = False
        record_start_time = None
        record_duration = self.record_duration  # 錄影秒數

        # 初始化 OpenCV 視窗
        # cv2.namedWindow("USB Camera", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("USB Camera", 640, 480)
        # cv2.resizeWindow("USB Camera", 1280, 960)

        

        # 取得攝影機畫面尺寸與 FPS
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 30.0
            print(f"無法取得原生 FPS，使用預設 FPS: {fps}")
        else:
            print(f"攝影機 FPS: {fps}")

        print("按下 'r' 錄製 30 秒，'q' 離開程式。")

        progress_bar = None  # 初始化進度條物件
        last_update_second = -1  # 控制 tqdm 更新頻率

        # frame_count = 0
        # start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取畫面")
                break
            
            calib_frame = cv2.undistort(frame.copy(), cameraMatrix=self.cameraMatrix, distCoeffs=self.distortion_coefficients)
            # calib_frame = frame.copy()
            cv2.imshow("USB Camera", frame)
            cv2.imshow("Calib Frame", calib_frame)
            yuv = cv2.cvtColor(calib_frame.copy(), cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)

            black_frame = np.zeros_like(frame, dtype=np.uint8)
            mask_u = (u != 128)
            mask_v = (v != 128)
            black_frame[mask_u] = 255
            black_frame[mask_v] = 255
            cv2.imshow("black_frame", black_frame)

            # cv2.imshow("y", y)
            # cv2.imshow("u", u)
            # cv2.imshow("v", v)
            # # 計算fps
            # frame_count += 1
            # elapsed_time = time.time() - start_time
            # if elapsed_time >= 1.0:
            #     fps = frame_count / elapsed_time
            #     print("FPS:", fps)
            #     return
            

            key = cv2.waitKey(1) & 0xFF

            # 如果按下 'r' 且尚未在錄影，開始錄製
            if key == ord('r') and not recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # filename = os.path.join(self.filepath, f"{timestamp}.mp4")
                # fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPEG 壓縮

                # filename = os.path.join(self.filepath, f"{timestamp}.avi")
                # fourcc = cv2.VideoWriter_fourcc(*'I420') # YUV2 壓縮
                                
                filename = os.path.join(self.filepath, f"{timestamp}.mkv")
                fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # 無損壓縮
                

                out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

                recording = True
                record_start_time = time.time()
                print(f"開始錄影：{filename}（{record_duration} 秒）")
                progress_bar = tqdm(total=record_duration, desc=filename, unit="秒", leave=False)
                

            # 如果正在錄影，持續寫入影片並檢查是否超過 30 秒
            if recording:
                out.write(frame)

                elapsed = time.time() - record_start_time
                elapsed_sec = int(elapsed)

                if elapsed_sec != last_update_second:
                    progress_bar.n = elapsed_sec
                    progress_bar.refresh()
                    last_update_second = elapsed_sec

                if elapsed >= record_duration:
                    recording = False
                    out.release()                    
                    progress_bar.n = record_duration
                    progress_bar.refresh()
                    progress_bar.close()
                    print("錄影結束並儲存。")
                    self.check_video(filename)

            # 離開程式
            if key == ord('q'):
                if recording:
                    out.release()
                    print("錄影結束並儲存。")
                break

        # 清除資源
        cap.release()
        cv2.destroyAllWindows()

    def check_video(self, filename):
        # 讀取影片檔案
        # video_path = os.path.join(filepath, filename)  # 這裡改成你的檔名
        cap = cv2.VideoCapture(filename)

        if not cap.isOpened():
            print("無法開啟影片")
        else:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0

            print("儲存為:", filename)
            print(f"影片總幀數：{frame_count}")
            print(f"FPS：{fps}")
            print(f"影片長度：{duration:.2f} 秒")

        cap.release()

    def replay_mvs(self, filename):
        # 讀取影片檔案
        video_path = os.path.join(self.filepath, filename)  # 這裡改成你的檔名
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("無法開啟影片")
            return


        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            yuv = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv) # y: 灰階, u: X軸向量, v: Y軸向量

            cv2.imshow("y", y)
            # cv2.imshow("u", u)
            # cv2.imshow("v", v)

            # 做一張同樣大小的黑色圖片
            black_frame = np.zeros_like(frame, dtype=np.uint8)
            # 如果 u != 128 或 v != 128，則將對應的像素設為白色
            mask_u = (u != 128)
            mask_v = (v != 128)
            black_frame[mask_u] = 255
            black_frame[mask_v] = 255
            cv2.imshow("black_frame", black_frame)

            # 依照fps播放影片
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                delay = int(1000 / fps)
            else:
                delay = 30

            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    videoRecorder = VideoRecorder()
    videoRecorder.record()
    # videoRecorder.replay_mvs("test_mvs_compressed.mp4") # ROSBAG 轉成mp4
    # videoRecorder.replay_mvs("20250416_150545.mp4") # MJPG
    # videoRecorder.replay_mvs("20250416_150325.mkv") # 無損壓縮
    # videoRecorder.replay_mvs("20250416_150048.avi") # YUV2
