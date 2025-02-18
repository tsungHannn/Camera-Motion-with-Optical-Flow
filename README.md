# MV on Vehicle
直接用 VS code 執行camera_motion_estimation.py
視窗介紹：
- polygons：執行結果，包含車道線檢測。
- average_v：X軸向量經過模糊化後的結果。
- gray_with_line：在ROI裡面檢測到的線段(車道檢測用，所以已經去除過短的、水平的線)


# rosbag2video.py
指令
```bash
python rosbag2video.py 資料夾名稱
```
會在相同資料夾內產生.mp4檔案，但結束的時候要ctrl+C。
