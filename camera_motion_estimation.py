import os

import cv2 as cv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import yaml
from ultralytics import YOLO
from utils import KalmanFilter
# from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from hdbscan import HDBSCAN
# from cuml.common.device_selection import using_device_type
# from cuml.cluster import DBSCAN as cuDBSCAN
# from cuml.cluster import HDBSCAN

# matplotlib.use("Qt5Agg")

"""
MVS 資料格式：
輸出為640*480 YUV422, 其中
Y: 灰階影像 or Edge(640*480)
U: 水平向量(320*480)  U > 128:向左   U < 128:向右
V: 垂直向量(320*480)  V > 128:向上   V < 128:向下

"""

class MV_on_Vechicle:
    def __init__(self):
        # 讀MV內參
        with open("mvs.yaml", "r") as file:
            mvs_data = yaml.load(file, Loader=yaml.FullLoader)
            self.cameraMatrix = np.array(mvs_data['camera_matrix']['data'])
            self.cameraMatrix = self.cameraMatrix.reshape(3,3)
            self.distortion_coefficients = np.array(mvs_data['distortion_coefficients']['data'])
            self.distortion_coefficients = self.distortion_coefficients.reshape(1,5)
        self.window_number = 40
        self.threshold = 8000
        self.frame_width = 640
        self.frame_height = 480


		# specify directory and file name
        # self.dir_path = "mvs_mp4\\0521"
        self.dir_path = "/media/mvclab/HDD/mvs_mp4/0701/edge"  # mvclab
        # self.all_file = os.listdir(self.dir_path)
        # self.all_file = sorted(self.all_file)
        # self.all_file = ["test_2024-03-18-07-57-26_mvs_compressed.mp4"] # 0318
        # self.all_file = ["test_2024-05-21-08-08-41_mvs_compressed.mp4"] # 0521
        self.all_file = ["test_2024-07-01-02-45-59_mvs_compressed.mp4"] # 0701
        # self.all_file = ["test_2024-06-28-10-11-20_mvs.mp4"]

  
        # set parameters for text drawn on the frames
        self.font = cv.FONT_HERSHEY_COMPLEX
        self.fontScale = 1
        self.fontColor = (68, 148, 213)
        self.lineType  = 3

        # 左右轉數值邊界
        self.leaning_right = 24
        self.leaning_left = 16

        self.window_list = [] # 存每個window(左右)
        self.yolo_window_list = []
        self.window_state = [] # 每個window的區域結果
        self.yolo_window_state = []
        self.polygon_list = [] # 畫每個window的範圍
        self.window_result = [] # 存每個window的結果
        self.yolo_window_result = []
        self.record = []
        self.last_state = [] # 方向點數量沒超過閥值的話就使用上一次的結果
        self.yolo_last_state = []
        self.center_list = [] # 紀錄中央點，畫圖用
        self.yolo_center_list = []
        self.center_without_avg_list = []
        self.yolo_cener_without_avg_list = []

        self.last_center = 20
        self.is_detect = True # 轉彎時不進行物件偵測



        self.model = YOLO('yolov8m.pt')
        
    # estimate left or right
    def lr_estimate(self, img):
        self.threshold = img.shape[0]*img.shape[1]//3
        
        translation = np.ravel(img) # 把img變為一維


        right_index = np.where(translation < 128)
        left_index = np.where(translation > 128)
    
        diff = len(right_index[0]) - len(left_index[0])
        if diff > self.threshold:
            return 1 # 向右
        elif diff < -1 * self.threshold:
            return -1 # 向左
        else:
            return "None"


    def find_center(self, arr):
        best_index = -1
        ans_list = []
        for i in range(len(arr)):
            if i == 0:
                left = 0
                right = np.sum(arr)
            elif i == len(arr) - 1:
                left = np.sum(arr)
                right = 0
            else:
                left = np.sum(arr[:i])
                right = np.sum(arr[i:])

            ans_list.append(abs(left-right))
        
        best_index = np.argmax(ans_list)

        if best_index == 0:
            if np.sum(arr) < 0:
                best_index = len(arr) - 1
            elif np.sum(arr) > 0:
                best_index = 0

        return best_index
    
    # 判斷偵測框內的 MV 是否正在超車
    # 輸入整張圖片及偵測框對角點
    def is_overtake(self, img, x1, y1, x2, y2):
        det_center = (x1+x2) // 2
        box = img[int(y1):int(y2), int(x1):int(x2)]
        threshold = box.shape[0]*box.shape[1] // 15

        translation = np.ravel(box) # 把img變為一維
        right_index = np.where(translation < 128)
        left_index = np.where(translation > 128)
        diff = len(right_index[0]) - len(left_index[0])

        if det_center < self.frame_width // 4:  # 框框在整個畫面的左邊
            if diff > threshold:    # 框框在左邊，而且框框內的mv值往右 -> 框框內的車是超車
                return True
            else:
                return False
        elif det_center > self.frame_width // 4*3:    # 框框在整個畫面的右邊
            if diff < -1 * threshold:    # 框框在右邊，而且框框內的mv值往左
                return True
            else:
                return False

    # 讀取 x 軸向量，若在正常直走時有雜點就消除
    # 會修改 v 的值
    def filter_overtake(self, img):
        middle_index = self.frame_width // 2

        # 左半部分處理
        left_half = img[:, :middle_index]
        left_half[left_half < 128] = 128

        # 右半部分處理
        right_half = img[:, middle_index:]
        right_half[right_half > 128] = 128

        # 合併處理後的圖像
        processed_image_array = np.hstack((left_half, right_half))

        return processed_image_array
    
    def dbscan_filter(self, img):

        middle_index = self.frame_width // 2

        # 正常來說左邊是白色點(<128)，所以只留大於128的點，再判斷是不是雜訊
        # 左半部分處理
        left_half = img[:, :middle_index]
        left_half[left_half > 128] = 128

        # 右半部分處理
        right_half = img[:, middle_index:]
        right_half[right_half < 128] = 128

        processed_image_array = np.hstack((left_half, right_half))

        # 正規化
        normalized_image_array = processed_image_array / 255.0
        mean = np.mean(processed_image_array)
        std = np.std(processed_image_array)
        standardized_image_array = (processed_image_array - mean) / std

        X = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # 將(i, j)座標和灰階值結合成一個數據點

                X.append([i, j, standardized_image_array[i][j]])


        
        cv.imshow("processed_img", processed_image_array)
        # 將數據點轉換為NumPy陣列
        X = np.array(X)
        # with using_device_type("GPU"):
            # db = HDBSCAN(min_samples=10).fit(X)
        db = HDBSCAN().fit(X)
        labels = db.labels_


        # 將分群結果轉換為可視化圖像

        # 創建一個與原圖相同尺寸的空白圖像，用於存放可視化結果
        clustered_visual = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        # 為每個群組分配一個隨機顏色（-1表示雜訊，會以黑色顯示）
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('hsv', len(unique_labels))(range(len(unique_labels)))
        colors = (colors[:, :3] * 255).astype(int)  # HSV to RGB and scale to 0-255

        # 遍歷每個點並分配顏色
        index = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                label = labels[index]
                if label == -1:
                    color = (0, 0, 0)  # 雜訊顯示為黑色
                else:
                    color = colors[label]
                clustered_visual[i, j] = color  # 為每個像素賦予RGB顏色值
                index += 1



        # 顯示或保存結果
        cv.imshow('Clustered Video', clustered_visual)
        # outputStream2.write(clustered_visual)


    def run(self):
        for file in self.all_file:
            filename = file


            # initialise stream from video
            cap = cv.VideoCapture(os.path.join(self.dir_path, filename))
            # cap = cv.VideoCapture(filename)
            print("-"*20)
            print(os.path.join(self.dir_path, filename))
            print(cap.isOpened())
            ret, prvs = cap.read()

            # initialise video writer
            frameRate = int(cap.get(cv.CAP_PROP_FPS))
            codec = cv.VideoWriter_fourcc(*'mp4v')
            save_name = "motion_" + filename[:-4] + ".mp4"
            outputStream1 = cv.VideoWriter(save_name, codec, frameRate, (int(cap.get(3)),int(cap.get(4))))
            outputStream2 = cv.VideoWriter("save.mp4", codec, frameRate, (int(cap.get(3)),int(cap.get(4))))


            # initialise text variables to draw on frames
            # motion_list = []
            # realMotion = 'None'


            frame_id = 0
            self.frame_width = int(cap.get(3))
            self.frame_height = int(cap.get(4))
            window_width = self.frame_width // self.window_number

            # 前面的frame_height-(frame_height//10 * 2)是為了不要底下雨刷的部份，不要底下的1/5範圍
            # window_height = (frame_height-int(frame_height//2)) // self.ud_window_number
            window_left = self.frame_width // 4
            window_right = self.frame_width // 4 * 3
            # window_left = 0
            # window_right = frame_width

            window_bottom = self.frame_height // 5 * 4
            window_top = self.frame_height // 5


            for i in range(self.window_number):
                self.last_state.append("")
                self.window_result.append([])
                self.yolo_last_state.append("")
                self.yolo_window_result.append([])
            

            stateMatrix = np.array([[20.0]], dtype=np.float32)

            estimateCovariance = np.array([[1.0]], dtype=np.float32)
            transitionMatrix = np.array([[1.0]], dtype=np.float32)
            processNoiseCov = np.array([[0.01]], dtype=np.float32)

            measurementStateMatrix = np.array([[0.0]], dtype=np.float32)

            observationMatrix = np.array([[1.0]], dtype=np.float32)
            measurementNoiseCov = np.array([[3.0]], dtype=np.float32)

            kf1 = KalmanFilter(X=stateMatrix,
                    P=estimateCovariance,
                    F=transitionMatrix,
                    Q=processNoiseCov,
                    Z=measurementStateMatrix,
                    H=observationMatrix,
                    R=measurementNoiseCov)
            kf2 = KalmanFilter(X=stateMatrix,
                    P=estimateCovariance,
                    F=transitionMatrix,
                    Q=processNoiseCov,
                    Z=measurementStateMatrix,
                    H=observationMatrix,
                    R=measurementNoiseCov)
            

            u_plot1 = []
            u_plot2 = []

            # main loop 
            while True:
                # read a new frame
                ret, nxt = cap.read()

                if not ret:
                    break
                
                # cv.imshow("before", nxt)
                
                nxt = cv.undistort(nxt, cameraMatrix=self.cameraMatrix, distCoeffs=self.distortion_coefficients)
                # cv.imshow("after", nxt)
                # cv.imwrite("after.jpg", nxt)
                yuv = cv.cvtColor(nxt.copy(), cv.COLOR_RGB2YUV)

                y, u, v = cv.split(yuv) # 不知道為啥 v看起來才是水平向量


                translation = np.ravel(u.copy())
                up = np.where(translation > 128)
                down = np.where(translation < 128)

                count_up = len(translation[up])
                count_down = len(translation[down])
                diff = abs(count_up - count_down)
                u_plot1.append(diff)

                

                # dbscanV = v.copy()
                # self.dbscan_filter(dbscanV)


                yuv_with_polygons = nxt.copy()

                # # 如果目前在轉彎，則不要進行物件偵測輔助
                # if len(self.center_list) > 20:
                #     if self.center_list[-1] < self.window_number//4 or self.center_list[-1] > self.window_number//4*3:
                #         cv.circle(yuv_with_polygons, (10,10), 10, (0, 0, 255), -1)
                #         self.is_detect = False
                #     else:
                #         cv.circle(yuv_with_polygons, (10,10), 10, (0, 255, 0), -1)
                #         self.is_detect = True
                # # self.is_detect = True

                # # yolo 偵測
                # yoloPicture = cv.merge((y, y, y))
                # yoloV = v.copy()
                # yoloResult = self.model(yoloPicture, verbose=False)
                # for result in yoloResult:
                #     for box in result.boxes:
                #         cls = box.cls
                #         classID = cls.item()
                #         if classID == 2  or classID==3 or classID == 0 or classID == 7:    # 2:car; 3:motorcycle; 5:bus; 7:truck
                #             x1, y1, x2, y2 = box.xyxy[0]
                #             if self.is_detect:
                #                 if self.is_overtake(yoloV, x1, y1, x2, y2):
                #                     yoloV[int(y1):int(y2), int(x1):int(x2)] = 128   # 偵測框內的MV值不計算 (128是沒有向量)
                #                     # conf = box.conf
                #                     cv.rectangle(yuv_with_polygons, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                #                     # cv.putText(yoloPicture, f'{self.model.names[int(cls.item())]} {conf.item():.2f}', (int(x1), int(y1)-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                #                 # else:
                #                 #     conf = box.conf
                #                 #     cv.rectangle(yuv_with_polygons, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                #                 #     cv.putText(yuv_with_polygons, f'{self.model.names[int(cls.item())]} {conf.item():.2f}', (int(x1), int(y1)-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                #             # else:
                #             #     conf = box.conf
                #             #     cv.rectangle(yuv_with_polygons, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                #             #     cv.putText(yuv_with_polygons, f'{self.model.names[int(cls.item())]} {conf.item():.2f}', (int(x1), int(y1)-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # cv.imshow("yolo", yoloV)

                # cv.imwrite("yolo.jpg", yoloPicture)


                # # x 軸向量前處理
                # testV = self.filter_overtake(v.copy())
                # cv.imshow("testV", testV)


                self.window_list.clear()
                # self.yolo_window_list.clear()
                self.polygon_list.clear()
                self.window_state.clear()
                # self.yolo_window_state.clear()

                # 直切
                for i in range(self.window_number):
                    self.window_list.append(v[window_top:window_bottom, window_width*i:window_width*(i+1)])
                    # self.yolo_window_list.append(yoloV[window_top:window_bottom, window_width*i:window_width*(i+1)])
                    
                    # # 實際偵測範圍
                    # polygon = [[window_width*i, window_top], [window_width*(i+1), window_top], [window_width*(i+1),window_bottom], [window_width*i, window_bottom]]
                    # polygon = np.array([polygon], dtype=np.int32)
                    # self.polygon_list.append(polygon)

                    # 示意框
                    polygon = [[window_width*i, window_top-40], [window_width*(i+1), window_top-40], [window_width*(i+1),window_top -20], [window_width*i, window_top - 20]]
                    polygon = np.array([polygon], dtype=np.int32)
                    self.polygon_list.append(polygon)


                # 畫偵測區域(漸層)
                for i in range(self.window_number):
                    # Calculate blue channel value for gradient
                    blue_value = int(255 * (self.window_number - i) / self.window_number + 80)
                    red_value = int(255 * i / self.window_number + 80)
                    # Draw polygon with calculated color
                    color_bgr = (red_value, 30, blue_value)
                    yuv_with_polygons = cv.polylines(yuv_with_polygons, self.polygon_list[i], isClosed=True, color=color_bgr, thickness=2)
                



                # 如果左右點差距小於閥值，就使用上一次的結果
                for i in range(self.window_number):
                    tempAns = self.lr_estimate(self.window_list[i])
                    # yolo_tempAns = self.lr_estimate(self.yolo_window_list[i])

                    if(tempAns == "None"):
                        self.window_state.append(self.last_state[i])
                    if(tempAns != "None"):
                        self.last_state[i] = tempAns
                        self.window_state.append(tempAns)
                    
                    # if(yolo_tempAns == "None"):
                    #     self.yolo_window_state.append(self.yolo_last_state[i])
                    # if(yolo_tempAns != "None"):
                    #     self.yolo_last_state[i] = yolo_tempAns
                    #     self.yolo_window_state.append(yolo_tempAns)
                    
                    

                # 儲存每個window的結果
                tempRow = []
                for i in range(self.window_number):
                    # cv.putText(yuv_with_polygons, str(window_state[i]), (window_width*i+20, 100), self.font, self.fontScale, self.fontColor, self.lineType)
                    if(self.window_state[i] == ""):
                        self.window_result[i].append(20)
                        tempRow.append(20)
                    else:
                        self.window_result[i].append(int(self.window_state[i]))
                        tempRow.append(int(self.window_state[i]))
                
                # 儲存每個window的結果
                # yolo_tempRow = []
                # for i in range(self.window_number):
                #     # cv.putText(yuv_with_polygons, str(window_state[i]), (window_width*i+20, 100), self.font, self.fontScale, self.fontColor, self.lineType)
                #     if(self.yolo_window_state[i] == ""):
                #         self.yolo_window_result[i].append(0)
                #         yolo_tempRow.append(0)
                #     else:
                #         self.yolo_window_result[i].append(int(self.yolo_window_state[i]))
                #         yolo_tempRow.append(int(self.yolo_window_state[i]))

                lr_center = self.find_center(tempRow)



                
                # =================================
                # 測試把波峰拿掉
                tempWindow = []
                if len(u_plot1) > 20:
                    for i in range(1, 21):
                        tempWindow.append(u_plot1[-i])

                    peaks, _ = find_peaks(tempWindow, prominence=25000)
                    if 1 in peaks:
                        temp_lr_center = self.last_center
                        u_plot2.append(frame_id)
                    else:
                        self.last_center = lr_center
                        temp_lr_center = lr_center
                else:
                    for i in range(len(u_plot1)):
                        tempWindow.append(u_plot1[-i])
                        
                    peaks, _ = find_peaks(tempWindow, prominence=25000)
                    if 1 in peaks:
                        temp_lr_center = self.last_center
                    else:
                        self.last_center = lr_center
                        temp_lr_center = lr_center




                self.center_without_avg_list.append(lr_center)
                self.yolo_cener_without_avg_list.append(temp_lr_center)

                # yolo_lr_center = self.find_center(yolo_tempRow)
                # self.yolo_cener_without_avg_list.append(yolo_lr_center)


                # # # Kalman Filter
                # current_measurement = np.array([[lr_center]], dtype=np.float32)
                # current_prediction = kf2.predict()
                # corrected_state = int(kf2.correct(current_measurement))
                # cv.circle(yuv_with_polygons, ((self.frame_width*corrected_state//self.window_number)+(window_width//2), window_top-30), 6, (31,198,0), -1)

                # current_measurement = np.array([[yolo_lr_center]], dtype=np.float32)
                # current_prediction = kf1.predict()
                # corrected_state = int(kf1.correct(current_measurement))
                # cv.circle(yuv_with_polygons, ((self.frame_width*corrected_state//self.window_number)+(window_width//2), window_top-50), 6, (0,0,255), -1)
                
                
                # 20 幀後才開始算
                if len(self.center_without_avg_list) >= 20:
                    center_sum = 0
                    yolo_center_sum = 0

                    for i in range(1, 21):
                        center_sum += self.center_without_avg_list[-i]
                        yolo_center_sum += self.yolo_cener_without_avg_list[-i]

                    center_avg = int(center_sum / 20)
                    yolo_center_avg = int(yolo_center_sum / 20)
                    





                    # # 畫中心位置
                    self.center_list.append(center_avg)
                    cv.circle(yuv_with_polygons, ((self.frame_width*center_avg//self.window_number)+(window_width//2), window_top-30), 6, (31, 198, 0), -1)
                    
                    # 畫中心位置
                    self.yolo_center_list.append(yolo_center_avg)
                    cv.circle(yuv_with_polygons, ((self.frame_width*yolo_center_avg//self.window_number)+(window_width//2), window_top-50), 6, (0, 0, 255), -1)
                    # cv.circle(yuv_with_polygons, ((self.frame_width*yolo_center_avg//self.window_number)+(window_width//2), window_top-30), 6, (31, 198, 0), -1)


                    

                    # # # 移動平均 + Kalman Filter
                    # current_measurement = np.array([[lr_center_avg]], dtype=np.float32)
                    # current_prediction = kf1.predict()
                    # corrected_state = int(kf1.correct(current_measurement))
                    # cv.circle(yuv_with_polygons, ((self.frame_width*corrected_state//self.lr_window_number)+(window_width//2), window_top-70), 6, (255,0,0), -1)

                else:
                    self.center_list.append(20) # 為了畫圖合理，前面20幀沒有數據補0
                    # self.yolo_center_list.append(0)


                    
                # if(len(motion_list) >= 3):
                #     if(motion_list[0] == motion_list[1] and motion_list[1] == motion_list[2]):
                #         realMotion = motion_list[0]


                # cv.putText(yuv_with_polygons, str(realMotion), (260,400), self.font, self.fontScale, (0, 0, 255), self.lineType)
                cv.putText(yuv_with_polygons, str(frame_id), (610, 20), self.font, 0.5, (0, 0, 255), 1)

                y = cv.cvtColor(y, cv.COLOR_GRAY2BGR)
                # u = cv.cvtColor(u, cv.COLOR_GRAY2BGR)
                # v = cv.cvtColor(v, cv.COLOR_GRAY2BGR)

                

                cv.imshow("polygons", yuv_with_polygons)

                # outputV = cv.merge((y,y,y))
                # cv.imshow("gray", y)
                # cv.imshow("y", y)
                cv.imshow("u", u)
                cv.imshow("v", v)
                # cv.imwrite("gray.jpg", y)
                # cv.imwrite("v.jpg", v)
                # cv.imwrite("polygons.jpg", yuv_with_polygons)
                # for i in range(self.window_number):
                #     cv.imshow(str(i), self.lr_window_list[i])


                # 畫即時圖表
                # plt.clf()
                # translation = np.ravel(u.copy())
                # up = np.where(translation > 128)
                # down = np.where(translation < 128)

                # count_up = len(translation[up])
                # count_down = len(translation[down])
                # diff = abs(count_up - count_down)
                # u_plot1.append(diff)
                # peaks, _ = find_peaks(u_plot1, prominence=25000)
                
                # u_plot2.append(count_down)
                # plt.plot(peaks, np.array(u_plot1)[peaks], 'x')
                # plt.plot(u_plot1)
                # plt.plot(u_plot2, label="down")
                # plt.show()
                # plt.pause(0.0000001)
                # plt.ioff()
                
                

                # if cv.waitKey(25) & 0xFF == ord('q'):
                #     break
                # # 不用 YOLO 偵測的話會跑太快
                if cv.waitKey(1000//frameRate) & 0xFF == ord('q'):
                    break
                
                outputV = cv.merge((u,u,u))
                outputStream1.write(yuv_with_polygons)
                outputStream2.write(outputV)


                frame_id += 1

            peaks, _ = find_peaks(u_plot1, prominence=25000)
            plt.plot(u_plot1)
            plt.plot(peaks, np.array(u_plot1)[peaks], 'x')
            plt.plot(u_plot2, np.array(u_plot1)[u_plot2], 'o')

            outputStream1.release()
            outputStream2.release()



if __name__ == "__main__":

    detector = MV_on_Vechicle()
    detector.run()
    # plt.cla()

    # plt.plot()
    # plt.plot(detector.lr_center_list, label="mapped")

    plt.xlabel("Frame", {'fontsize':20})
    plt.ylabel("Gap", {'fontsize':20})
    # plt.legend(
    #     loc='best',
    #     fontsize=20,
    #     shadow=True,
    #     facecolor='#ccc',
    #     edgecolor='#000',
    #     title='test',
    #     title_fontsize=20)
    plt.savefig("plot.png")
    plt.show()
    # MV_on_Vechicle().run_split_window()
    # MV_on_Vechicle().run_two_window()
