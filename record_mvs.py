import cv2
import time
from datetime import datetime


def record():
    # 初始化攝影機 (你使用的是 index 1 並指定 DirectShow backend)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    # 確保攝影機成功開啟
    if not cap.isOpened():
        print("無法開啟攝影機")
        exit()

    # 初始化錄影器
    out = None
    recording = False
    record_start_time = None
    record_duration = 30  # 錄影秒數

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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取畫面")
            break

        cv2.imshow("USB Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        # 如果按下 'r' 且尚未在錄影，開始錄製
        if key == ord('r') and not recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
            recording = True
            record_start_time = time.time()
            print(f"開始錄影：{filename}（30 秒）")

        # 如果正在錄影，持續寫入影片並檢查是否超過 30 秒
        if recording:
            out.write(frame)
            if time.time() - record_start_time >= record_duration:
                recording = False
                out.release()
                print("錄影結束並儲存。")

        # 離開程式
        if key == ord('q'):
            if recording:
                out.release()
                print("錄影結束並儲存。")
            break

    # 清除資源
    cap.release()
    cv2.destroyAllWindows()


def check_video():
    # 讀取影片檔案
    video_path = "20250414_144748.mp4"  # 這裡改成你的檔名
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("無法開啟影片")
    else:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0

        print(f"影片總幀數：{frame_count}")
        print(f"FPS：{fps}")
        print(f"影片長度：{duration:.2f} 秒")

    cap.release()


if __name__ == "__main__":
    record()
    # check_video()