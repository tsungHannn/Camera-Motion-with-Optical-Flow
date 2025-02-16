# 讀圖片後點擊，可以得到點擊的像素座標


import cv2

# 滑鼠點擊事件處理函式
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 左鍵點擊事件
        print(f"滑鼠點擊座標：({x}, {y})")

# 讀取圖片
image_path = "gray.jpg"  # 替換成你的圖片路徑
image = cv2.imread(image_path)

if image is None:
    print("無法讀取圖片，請確認圖片路徑是否正確")
    exit()

# 創建窗口並顯示圖片
cv2.namedWindow("Image Viewer")
cv2.setMouseCallback("Image Viewer", mouse_callback)

while True:
    cv2.imshow("Image Viewer", image)
    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
