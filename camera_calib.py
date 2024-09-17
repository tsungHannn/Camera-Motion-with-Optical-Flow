import cv2
import numpy as np
import glob

def calibrate_camera(images_folder, chessboard_size=(9, 6), square_size=29):
    # 設置棋盤格大小和真實世界每個方格的大小
    chessboard_size = chessboard_size
    square_size = square_size

    # 準備對象點，例如：(0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 用於存儲對象點和圖像點的數組
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # 獲取圖像文件列表
    images = glob.glob(f'{images_folder}/*.png')

    # 檢查是否有有效圖像
    if not images:
        raise ValueError("未找到圖像，請檢查圖像路徑和文件名是否正確。")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 尋找棋盤格角點
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # 如果找到，添加對象點和圖像點
        if ret:
            objpoints.append(objp)

            # 細化角點
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

            # 繪製角點並顯示
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.namedWindow("Chessboard", 0)
            cv2.imshow('Chessboard', img)

            # # 等待按下 'q' 鍵繼續
            # while True:
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
        else:
            print(f"未能在圖像 {fname} 中找到棋盤格角點，請檢查該圖像是否包含完整的棋盤格。")

    cv2.destroyAllWindows()

    # 確保有足夠的圖像用於校正
    if len(objpoints) == 0 or len(imgpoints) == 0:
        raise RuntimeError("未能找到足夠的棋盤格角點進行校正，請檢查輸入的圖像集。")

    # 進行相機校正
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

# 設置棋盤格圖像文件夾路徑
images_folder = 'pic/'

# 校正相機並獲取內參
camera_matrix, dist_coeffs = calibrate_camera(images_folder)

print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)
