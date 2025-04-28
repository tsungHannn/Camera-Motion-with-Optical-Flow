import cv2
import os
import numpy as np

def save_video_frames(video_path, output_folder):
    # 確認輸出資料夾存在，若不存在則建立
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 如果資料夾已存在，則清空資料夾中的所有檔案
    else:
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"無法刪除檔案 {file_path}: {e}")

    # 讀取影片
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break  # 影片讀取結束

        # 儲存當前幀為圖片
        frame_filename = os.path.join(output_folder, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(frame_filename, frame)

        print(f"已儲存: {frame_filename}")

        frame_idx += 1

    cap.release()
    print("影片幀儲存完成！")


def split_and_display_images(image_folder):
    # 取得資料夾內所有圖片檔案名稱，並排序
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print("資料夾中沒有找到圖片！")
        return

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path)

        if img is None:
            print(f"無法讀取圖片: {image_path}")
            continue
        
        # 轉成 YUV 色彩空間
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y_channel, u_channel, v_channel = cv2.split(yuv)
        # 分離出三個通道
        # b_channel, g_channel, r_channel = cv2.split(img)

        black_frame = np.zeros_like(y_channel, dtype=np.uint8)
        mask_u = (u_channel != 128)
        mask_v = (v_channel != 128)
        black_frame[mask_u] = 255
        black_frame[mask_v] = 255
        cv2.imshow("black_frame", black_frame)
        # 顯示原圖與各個通道
        cv2.imshow("Original", img)
        cv2.imshow("Y", y_channel)
        cv2.imshow("U", u_channel)
        cv2.imshow("V", v_channel)

        print(f"正在顯示: {image_file}")

        key = cv2.waitKey(0)  # 按任意鍵繼續到下一張
        if key == 27:  # 如果按下 ESC 鍵就離開
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":

    video_path = os.path.join("crowd_mp4", "20250427_143938.mkv")  # 這裡填你的影片檔路徑
    output_folder = "crowd_pic"         # 這裡填想儲存的資料夾

    # save_video_frames(video_path, output_folder) # 影片轉圖片
    split_and_display_images(output_folder)  # 圖片分離顯示
