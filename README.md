# MV on Vehicle
直接用 VS code 執行camera_motion_estimation.py\
目前是把車道檢測也放在camera_motion_estimation裡面

視窗介紹：
- polygons：執行結果，包含車道線檢測。
- average_v：X軸向量經過模糊化後的結果。
- gray_with_line：在ROI裡面檢測到的線段(車道檢測用，所以已經去除過短的、水平的線)

步驟:
1. 濾除相機晃動(檢測到晃動直接捨棄該幀)
2. 濾除雜訊(模糊化: cv.blur, kernel_size = 9)
3. 切Window
4. 計算每個Window的方向
5. 找中心位置
6. 平滑結果(移動平均+Kalman filter)

# 車道線檢測
lane_detection.py\
步驟:
1. 線段檢測(Canny)
2. 畫ROI
3. 過濾水平線
4. 分別計算斜率<0 跟 斜率>0的平均線段，作為最終車道線
5. 計算兩條車道線交點



# rosbag2video.py
指令
```bash
python rosbag2video.py 資料夾名稱
```
會在相同資料夾內產生.mp4檔案，但結束的時候要ctrl+C。
