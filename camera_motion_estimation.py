import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2 as cv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import yaml
from ultralytics import YOLO
from utils import KalmanFilter
# from sklearn.cluster import DBSCAN
# from sklearn.cluster import HDBSCAN
# from hdbscan import HDBSCAN
# from cuml.common.device_selection import using_device_type
# from cuml.cluster import DBSCAN as cuDBSCAN
# from cuml.cluster import HDBSCAN
import time

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
            
            
        
        with open('mvs.yaml', 'r') as file:
            self.mvs_config_lines = file.readlines()
        

        
        self.window_number = 40
        self.threshold = 8000
        self.frame_width = 640
        self.frame_height = 480


		# specify directory and file name
        # self.dir_path = "mvs_mp4\\1220\\translation"
        self.dir_path = "/media/mvclab/HDD/mvs_mp4/0701/gray"  # mvclab
        self.all_file = os.listdir(self.dir_path)
        self.all_file = sorted(self.all_file)
        # print("all_file:", self.all_file)
        # self.all_file = ["test_2024-03-18-07-57-26_mvs_compressed.mp4"] # 0318
        # self.all_file = ["test_2024-05-21-08-08-41_mvs_compressed.mp4"] # 0521
        # self.all_file = ["test_2024-07-01-02-38-53_mvs_compressed.mp4"] # 0701 edge
        self.all_file = ["test_2024-07-01-02-33-02_mvs_compressed.mp4"] # 0701 gray
        # self.all_file = ["2024-11-08-03-32-21_mvs_compressed.mp4"] # 1108 edge
        # self.all_file = ["2024-12-20-06-36-00_mvs_compressed.mp4"] # 1220
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
        self.comp_window_list = []
        self.window_state = [] # 每個window的區域結果
        self.comp_window_state = []
        self.polygon_list = [] # 畫每個window的範圍
        self.last_state = [] # 方向點數量沒超過閥值的話就使用上一次的結果
        self.state_buffer = [] # 紀錄之前所有有資訊的window
        self.comp_last_state = []
        self.center_list = [] # 紀錄中央點，畫圖用
        self.comp_center_list = []
        self.center_without_avg_list = []
        self.comp_center_without_avg_list = []


        self.record_window_list = [] # 把每一幀window_list存起來，作為相機位置校正用
        # record_window_list會有(frame_number, window_number)個
        # 如果影片共有720幀，切成40個window，record_window_list就會是720*40的二維陣列

        self.comp_last_state_time = [] # 紀錄這個window有幾次沒有資訊
        for i in range(self.window_number):
            self.comp_last_state_time.append(0)
            self.state_buffer.append([])
            self.last_state.append("")
            self.comp_last_state.append("")

        self.calibration_mode = False   # 校正模式，True時開始記錄中心點，直到程式結束或再度按下c
        # MVS 校正: 如果MVS放歪，偵測結果會歪掉。透過直走一段距離來抓正中間，調整向右、向左的offset讓直走時中心點在正中間
        self.turning_offset = 0

        self.last_center = 20
        self.is_detect = True # 轉彎時不進行物件偵測
        self.last_v = None # 如果偵測到y軸波峰，就把整個x軸向量換成上一幀


        # self.model = YOLO('yolov8m.pt')
        
    # estimate left or right
    def estimate(self, img):
       
        
        translation = np.ravel(img) # 把img變為一維
        

        right_index = np.where(translation < 128)
        left_index = np.where(translation > 128)

        

        self.threshold = img.shape[0]*img.shape[1] * 0.4

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
    


    # estimate left or right
    def calib_estimate(self, img, weight):
       
        
        translation = np.ravel(img) # 把img變為一維
        
        right_index = np.where(translation < 128)
        left_index = np.where(translation > 128)

        
        self.threshold = img.shape[0]*img.shape[1] * 0.4

        diff = len(right_index[0]) - len(left_index[0])


        if diff > self.threshold:
            return 1 # 向右
        elif diff < -1 * self.threshold:
            return -1 * weight # 向左
        else:
            return "None"
        
    # 相機如果擺歪，偵測到的方向會偏向某一邊。
    def camera_position_calibration(self):
        data = np.load("frames.npz", allow_pickle=True)
        loaded_frames = data["frames"]
        record_score = {}

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
        
        for weight in np.arange(0.1, 2.1, 0.1):
            
            last_state = []
            center_without_avg_list = []
            current_score = 0

            for i in range(self.window_number):
                last_state.append("")

            for frame in loaded_frames:
                    
                window_state = []
                # 如果左右點差距小於閥值，就使用上一次的結果
                for i in range(self.window_number):
                    tempAns = self.calib_estimate(frame[i], round(weight, 1))

                    if(tempAns != "None"):
                        last_state[i] = tempAns
                        window_state.append(tempAns)
                    elif(tempAns == "None"):
                        window_state.append(last_state[i])



                # 儲存每個window的結果
                tempRow = []
                for i in range(self.window_number):
                    # cv.putText(yuv_with_polygons, str(window_state[i]), (window_width*i+20, 100), self.font, self.fontScale, self.fontColor, self.lineType)
                    if(window_state[i] == ""):
                        tempRow.append(0)
                    else:
                        tempRow.append(window_state[i])


                lr_center = self.find_center(tempRow)
                center_without_avg_list.append(lr_center)

                # 卡爾曼濾波
                center_measurement = np.array([[lr_center]], dtype=np.float32)
                current_prediction = kf1.predict()
                center_kf1 = int(kf1.correct(center_measurement))


                # cv.circle(yuv_with_polygons, ((self.frame_width*corrected_state//self.window_number)+(window_width//2), window_top-50), 6, (0,0,255), -1)
                
                
                # 20 幀後才開始算
                if len(center_without_avg_list) >= 20:
                    center_sum = 0

                    for i in range(1, 21):
                        center_sum += center_without_avg_list[-i]

                    center_avg = int(center_sum / 20)
                    


                    center_avg = int((center_avg + center_kf1) / 2) # 卡爾曼濾波 + 移動平均

                    if center_avg == (self.window_number / 2) or center_avg == (self.window_number / 2 - 1): # 最中間
                        current_score += 5
                    elif center_avg == (self.window_number / 2 + 1) or center_avg == (self.window_number / 2 - 2): # 旁邊一格
                        current_score += 4
                    elif center_avg == (self.window_number / 2 + 2) or center_avg == (self.window_number / 2 - 3): # 旁邊兩格
                        current_score += 3
                    elif center_avg == (self.window_number / 2 + 3) or center_avg == (self.window_number / 2 - 4): # 旁邊三格
                        current_score += 2
                    elif center_avg == (self.window_number / 2 + 4) or center_avg == (self.window_number / 2 - 5): # 旁邊四格
                        current_score += 1



                    # # 把沒有資訊的window遮住
                    # for i in range(self.window_number):
                    #     if self.comp_last_state_time[i] > 10:
                    #         cv.circle(yuv_with_polygons, ((self.frame_width*i//self.window_number)+(window_width//2), window_top-30), 6, (0,0,0), -1)

                
                    # 畫中心位置
                    # cv.circle(yuv_with_polygons, ((self.frame_width*center_avg//self.window_number)+(window_width//2), window_top-30), 6, (31, 198, 0), -1)
                    # cv.circle(yuv_with_polygons, ((self.frame_width*comp_center_avg_after_calib//self.window_number)+(window_width//2), window_top-50), 6, (0, 0, 255), -1)
                
            # 紀錄每個weight的分數
            record_score[round(weight, 1)] = current_score

        
        count = 0
        for i in record_score:
            count += 1
            print(i, record_score[i])
        
        if count != 20:
            print()
        

   
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
            outputResult = cv.VideoWriter(save_name, codec, frameRate, (int(cap.get(3)),int(cap.get(4))))
            outputStream1 = cv.VideoWriter("output1.mp4", codec, frameRate, (int(cap.get(3)),int(cap.get(4))))
            outputStream2 = cv.VideoWriter("output2.mp4", codec, frameRate, (int(cap.get(3)),int(cap.get(4))))
            outputStream3 = cv.VideoWriter("output3.mp4", codec, frameRate, (int(cap.get(3)),int(cap.get(4))))
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


            # for i in range(self.window_number):
            #     self.last_state.append("")
            #     self.comp_last_state.append("")
            

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
            # u_plot2 = []
            y_peak_window = []
            v_window = []
            # plt.ion()

            ignored_frame = 0
            # main loop 
            start_time = time.time()
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
                
                v_window.insert(0, v)
                if len(v_window) <= 3:
                    continue
                if len(v_window) > 10:
                    v_window.pop()
                

                # 如果y軸檢測出波峰，就捨棄本次結果，用上一幀
                translation = np.ravel(u.copy())
                average = np.mean(translation)
                # u_plot1.append(average)

                # =================================
                # 儲存前50幀，以作為判斷波峰的依據
                y_peak_window.insert(0, average)
                # tempWindow.append(u_plot1[-1])
                if len(y_peak_window) > 50:
                    y_peak_window.pop()
                
                np_y_peak_window = np.array(y_peak_window.copy())
                peaks, _ = find_peaks(np_y_peak_window, prominence=1.5)
                valley, _ = find_peaks(-np_y_peak_window, prominence=1.5)

                # if len(peaks) > 0 or len(valley) > 0:
                #     print()
                for i in [2, 1, 0]:
                    if i in peaks or i in valley:
                        # print("Ignore", i, "FrameID", frame_id)
                        ignored_frame += 1
                        v_window[i] = v_window[i+1]

                

                # dbscanV = v.copy()
                # filter_vis, filter_v = self.dbscan_filter(dbscanV)


                yuv_with_polygons = nxt.copy()




                self.window_list.clear()
                self.comp_window_list.clear()
                self.polygon_list.clear()
                self.window_state.clear()
                self.comp_window_state.clear()

                # median_v = v_window[3].copy()
                # median_v = cv.medianBlur(median_v, 9)
                average_v = v_window[3].copy()
                average_v = cv.blur(average_v, (9, 9))
                # median_v = cv.GaussianBlur(median_v, (9,9), 0)
                # cv.imshow("median_v", median_v)
                # cv.imshow("average_v", average_v)

                # 視覺化v_window[3]跟median_v的差別
                # diff_v = average_v.copy()
                # diff_v[:] = 0
                # difference = np.abs(v_window[3].astype(np.int16) - average_v.astype(np.int16))
                # diff_mask = difference > 20
                # diff_v[diff_mask] = 255
                # diff_v[:] = 128
                # diff_mask = (v_window[3] > 128) & (average_v < 128) # 向左變向右
                # diff_v[diff_mask] = 255
                # diff_mask = (v_window[3] < 128) & (average_v > 128) # 向右變向左
                # diff_v[diff_mask] = 255
                # diff_mask = (v_window[3] != 128) & (average_v == 128) # 變成128
                # diff_v[diff_mask] = 255
                # diff_mask = (v_window[3] == 128) & (average_v != 128) # 變成不是128
                # diff_v[diff_mask] = 255
                # cv.imshow("diff", diff_v)
                # 直切
                for i in range(self.window_number):
                    # 經過y軸檢測波峰後，開始檢測移動方向
                    # v_window[3]是因為前兩幀用來檢測波峰
                    self.window_list.append(average_v[window_top:window_bottom, window_width*i:window_width*(i+1)])
                    self.comp_window_list.append(average_v[window_top:window_bottom, window_width*i:window_width*(i+1)])
                    
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
                
                
                # 紀錄每個window_list，給相機位置校正用
                self.record_window_list.append(self.window_list.copy())


                # 如果左右點差距小於閥值，就使用上一次的結果
                for i in range(self.window_number):
                    tempAns = self.estimate(self.window_list[i])
                    comp_tempAns = self.estimate(self.comp_window_list[i])

                    # tempAns = self.calib_estimate(self.window_list[i], 1)
                    # comp_tempAns = self.calib_estimate(self.comp_window_list[i], 1.9)

                    
                    if(tempAns != "None"):
                        self.last_state[i] = tempAns
                        self.window_state.append(tempAns)
                    elif(tempAns == "None"):
                        self.window_state.append(self.last_state[i])

                    if(comp_tempAns != "None"):
                        self.comp_last_state[i] = comp_tempAns
                        self.comp_window_state.append(comp_tempAns)
                        # self.comp_last_state_time[i] = 0

                        # self.state_buffer[i].append(comp_tempAns)
                        # if len(self.state_buffer[i]) > 5:
                        #     self.state_buffer[i].pop(0)

                    elif(comp_tempAns == "None"):
                        # 10 幀沒有資訊就不繼續沿用上一幀的結果
                        # self.comp_last_state_time[i] += 1
                        # if self.comp_last_state_time[i] > 10:
                        #     self.comp_window_state.append("")
                        # else:
                        #     self.comp_window_state.append(self.comp_last_state[i])

                        # # 沒有資訊的化，使用buffer的結果投票。buffer是紀錄之前有資訊的window
                        # if len(self.state_buffer[i]) == 5:
                        #     if self.state_buffer[i].count(1) > self.state_buffer[i].count(-1):
                        #         self.comp_window_state.append(1)
                        #     elif self.state_buffer[i].count(1) < self.state_buffer[i].count(-1):
                        #         self.comp_window_state.append(-1)
                        #     else:
                        #         print("Error")
                        # # buffer沒有5個先前資訊就用上一次有資訊的結果
                        # else:
                        #     self.comp_window_state.append(self.comp_last_state[i])


                        # 沒有資訊就用上一次有資訊的結果
                        self.comp_window_state.append(self.comp_last_state[i])


                    
                    

                # 儲存每個window的結果
                tempRow = []
                for i in range(self.window_number):
                    # cv.putText(yuv_with_polygons, str(window_state[i]), (window_width*i+20, 100), self.font, self.fontScale, self.fontColor, self.lineType)
                    if(self.window_state[i] == ""):
                        tempRow.append(0)
                    else:
                        tempRow.append(int(self.window_state[i]))
                
                # 儲存每個window的結果
                comp_tempRow = []
                for i in range(self.window_number):
                    # cv.putText(yuv_with_polygons, str(window_state[i]), (window_width*i+20, 100), self.font, self.fontScale, self.fontColor, self.lineType)
                    if(self.comp_window_state[i] == ""):
                        comp_tempRow.append(0)
                    else:
                        comp_tempRow.append(self.comp_window_state[i])


                lr_center = self.find_center(tempRow)
                self.center_without_avg_list.append(lr_center)

                comp_lr_center = self.find_center(comp_tempRow)
                self.comp_center_without_avg_list.append(comp_lr_center)
                

                
                # self.yolo_cener_without_avg_list.append(temp_lr_center)
               


                # # # Kalman Filter
                # current_measurement = np.array([[lr_center]], dtype=np.float32)
                # current_prediction = kf2.predict()
                # corrected_state = int(kf2.correct(current_measurement))
                # cv.circle(yuv_with_polygons, ((self.frame_width*corrected_state//self.window_number)+(window_width//2), window_top-30), 6, (31,198,0), -1)

                center_measurement = np.array([[lr_center]], dtype=np.float32)
                current_prediction = kf1.predict()
                center_kf1 = int(kf1.correct(center_measurement))

                comp_center_measurement = np.array([[comp_lr_center]], dtype=np.float32)
                current_prediction = kf2.predict()
                comp_center_kf2 = int(kf2.correct(comp_center_measurement))

                # cv.circle(yuv_with_polygons, ((self.frame_width*corrected_state//self.window_number)+(window_width//2), window_top-50), 6, (0,0,255), -1)
                
                
                # 20 幀後才開始算
                if len(self.center_without_avg_list) >= 20:
                    center_sum = 0
                    comp_center_sum = 0

                    for i in range(1, 21):
                        center_sum += self.center_without_avg_list[-i]
                        comp_center_sum += self.comp_center_without_avg_list[-i]

                    center_avg = int(center_sum / 20)
                    comp_center_avg = int(comp_center_sum / 20)
                    


                    center_avg = int((center_avg + center_kf1) / 2) # 卡爾曼濾波 + 移動平均
                    comp_center_avg = int((comp_center_avg + comp_center_kf2) / 2)


                    # # 把沒有資訊的window遮住
                    # for i in range(self.window_number):
                    #     if self.comp_last_state_time[i] > 10:
                    #         cv.circle(yuv_with_polygons, ((self.frame_width*i//self.window_number)+(window_width//2), window_top-30), 6, (0,0,0), -1)

                    
                    # 相機位置校正
                    # 先計算目前位置到中間差多少格
                    
                    comp_center_avg_after_calib = comp_center_avg - self.turning_offset

                    # 線性插值 y = y_min + (x-x_min) * (y_max - y_min) / (x_max - x_min)
                    
                    if self.turning_offset > 0: # 向左移 → 值域會從負的到(40-offset)
                        # center會沒辦法到最右邊，因為值域只到(40-offset)
                        # 所以如果center在右側，要插值到[number/2, number-1]之間
                        if comp_center_avg_after_calib > self.window_number / 2:    
                            # y_min = number/2; y_max = number-1
                            # x_min = number/2; x_max = number-1-offset
                            comp_center_avg_after_calib = (self.window_number / 2) + (comp_center_avg_after_calib-self.window_number/2) * ((self.window_number-1) - (self.window_number/2)) / ((self.window_number-1 - self.turning_offset) - self.window_number/2)
                    
                    if self.turning_offset < 0: # 向右移 → 值域會從offset到(40+offset)
                        # center會沒辦法到最左邊，因為值域只到offset
                        # 所以如果center在左側，要插值到[0, number/2]之間
                        if comp_center_avg_after_calib < self.window_number / 2:
                            # y_min = 0; y_max = number/2
                            # x_min = 0-offset; x_max = number/2
                            comp_center_avg_after_calib = 0 + (comp_center_avg_after_calib - (0 - self.turning_offset)) * ((self.window_number/2) - 0) / ((self.window_number/2) - (0 - self.turning_offset))



                    if comp_center_avg_after_calib > self.window_number - 1:
                        comp_center_avg_after_calib = self.window_number - 1
                    if comp_center_avg_after_calib < 0:
                        comp_center_avg_after_calib = 0
                    comp_center_avg_after_calib = int(comp_center_avg_after_calib)

                    # 畫中心位置
                    cv.circle(yuv_with_polygons, ((self.frame_width*center_avg//self.window_number)+(window_width//2), window_top-30), 6, (31, 198, 0), -1)
                    cv.circle(yuv_with_polygons, ((self.frame_width*comp_center_avg_after_calib//self.window_number)+(window_width//2), window_top-50), 6, (0, 0, 255), -1)
                    
                    

                    if self.calibration_mode == True:
                        self.center_list.append(center_avg)
                    

                    

                # else:
                    # self.center_list.append(20) # 為了畫圖合理，前面20幀沒有數據補0
                    # self.comp_center_list.append(20)


                    
                # if(len(motion_list) >= 3):
                #     if(motion_list[0] == motion_list[1] and motion_list[1] == motion_list[2]):
                #         realMotion = motion_list[0]


                # cv.putText(yuv_with_polygons, str(realMotion), (260,400), self.font, self.fontScale, (0, 0, 255), self.lineType)

                if self.calibration_mode == True:
                    cv.circle(yuv_with_polygons, (10, 10), 5, (0, 0, 255), -1)
                cv.putText(yuv_with_polygons, str(frame_id), (590, 20), self.font, 0.5, (0, 0, 255), 1)

                y = cv.cvtColor(y, cv.COLOR_GRAY2BGR)
                # u = cv.cvtColor(u, cv.COLOR_GRAY2BGR)
                # v = cv.cvtColor(v, cv.COLOR_GRAY2BGR)

                

                cv.imshow("polygons", yuv_with_polygons)

                # outputV = cv.merge((y,y,y))
                # cv.imshow("gray", y)
                # cv.imshow("y", y)
                # cv.imshow("u", u)
                # cv.imshow("v", v)
                cv.imshow("average_v", average_v)
                cv.imwrite("gray.jpg", y)
                # cv.imwrite("v.jpg", v)
                # cv.imwrite("polygons.jpg", yuv_with_polygons)
                # for i in range(self.window_number):
                #     cv.imshow(str(i), self.lr_window_list[i])

                # # 畫即時圖表
                # plt.clf()
                # translation = np.ravel(u.copy())
                # up = np.where(translation > 128)
                # down = np.where(translation < 128)

                # # count_up = len(translation[up])
                # # count_down = len(translation[down])
                # # diff = abs(count_up - count_down)
                # # u_plot1.append(diff)
                # average_up = np.mean(translation[up])
                # u_plot2.append(average_up)
                # # peaks, _ = find_peaks(u_plot1, prominence=25000)
                
                # # u_plot2.append(count_down)
                # # plt.plot(peaks, np.array(u_plot1)[peaks], 'x')
                # # plt.plot(u_plot1, label="up")
                # # plt.plot(u_plot2, label="down")
                # # plt.show()
                # # plt.pause(0.0000001)
                # # plt.ioff()
                
                
                key = cv.waitKey(25)
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    if len(self.center_without_avg_list) >= 20:
                        self.turning_offset = comp_center_avg - (self.window_number / 2)
                        print("offset:", self.turning_offset)

                # if cv.waitKey(25) & 0xFF == ord('q'):
                #     break
                # # 不用 YOLO 偵測的話會跑太快
                # if cv.waitKey(1000//frameRate) & 0xFF == ord('q'):
                #     break
                
                # # calibration mode
                # if cv.waitKey() & 0xFF == ord('c'):
                #     if self.calibration_mode == True:
                #         self.calibration_mode = False
                #     elif self.calibration_mode == False:
                #         self.calibration_mode = True


                # outputV = cv.merge((v,v,v))
                outputResult.write(yuv_with_polygons)
                # outputStream1.write(outputV)
                # outputStream2.write(cv.merge((average_v, average_v, average_v)))
                # outputStream3.write(cv.merge((diff_v, diff_v, diff_v)))


                # if frame_id == 44:
                #     for i in range(40, 60):
                #         for j in range(80, 100):
                #             print(v[i][j], "\t", end="")
                #         print()
                    
                #     print()

                frame_id += 1

                
            # calib_center = np.average(self.center_list)
            # self.turning_offset = (calib_center - (self.window_number / 2)) / self.window_number

            # # 為了不修改mvs.yaml的格式，只能這樣修改瞜
            # for i, line in enumerate(self.mvs_config_lines):
            #     if 'offset:' in line:
            #         self.mvs_config_lines[i] = f'  offset: {self.turning_offset}\n'
            # # 寫回 YAML 文件
            # with open('test.yaml', 'w') as file:
            #     file.writelines(self.mvs_config_lines)

            # u_plot1 = np.array(u_plot1)
            # peaks, _ = find_peaks(u_plot1, prominence=2)
            
            # valleys, _ = find_peaks(-u_plot1, prominence=2)

            # plt.plot(u_plot1)
            # plt.plot(peaks, np.array(u_plot1)[peaks], 'x')
            # plt.plot(valleys, np.array(u_plot1)[valleys], 'o')
            # plt.plot(u_plot2, np.array(u_plot1)[u_plot2], 'o')


            outputResult.release()
            outputStream1.release()
            outputStream2.release()
            outputStream3.release()
            end_time = time.time()
            execution_time = end_time - start_time
            minutes = int(execution_time // 60)
            seconds = int(execution_time % 60)
            print(f"Execution time: {minutes}min {seconds}sec")
            print("Frame:", frame_id)
            print("Ignored Frame:", ignored_frame)
            np.savez_compressed("frames.npz", frames=self.record_window_list)
            # return



if __name__ == "__main__":

    detector = MV_on_Vechicle()
    detector.run()
    # detector.camera_position_calibration()
    # plt.cla()

    # plt.plot()
    # plt.plot(detector.lr_center_list, label="mapped")

    # plt.xlabel("Frame", {'fontsize':20})
    # plt.ylabel("Average", {'fontsize':20})
    # # plt.legend(
    # #     loc='best',
    # #     fontsize=20,
    # #     shadow=True,
    # #     facecolor='#ccc',
    # #     edgecolor='#000',
    # #     title='test',
    # #     title_fontsize=20)
    # plt.savefig("plot.png")
    # plt.show()
    # MV_on_Vechicle().run_split_window()
    # MV_on_Vechicle().run_two_window()
