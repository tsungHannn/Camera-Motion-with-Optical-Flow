import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import yaml
from ultralytics import YOLO
"""
MVS 資料格式：
輸出為640*480 YUV422, 其中
Y: 灰階影像 or Edge(640*480)
U: 水平向量(320*480)  U > 128:向左   U < 128:向右
V: 垂直向量(320*480)  V > 128:向上   V < 128:向下

"""


# 初始化卡爾曼濾波器的變量
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, estimated_measurement_variance):
        # 過程噪聲：該參數描述了系統本身的不確定性，也就是模型預測的噪聲大小。
        # 如果設置較高的過程噪聲方差，濾波器會更相信觀測值
        self.process_variance = process_variance

        # 測量噪聲：該參數描述了觀測值的不確定性，也就是測量噪聲的大小。
        # 如果測量噪聲方差較大，濾波器會對測量數據的信任度降低，更依賴於模型的預測，這可能使濾波器的響應更平滑，但對觀測數據的突然變化不敏感。
        self.measurement_variance = measurement_variance

        # 估計的測量誤差協方差:設定了濾波器開始運行時的初始狀態不確定性。
        # 初始值設置為較大時，濾波器會在初始幾步中迅速調整，以更快地收斂到正確的值。
        self.estimated_measurement_variance = estimated_measurement_variance
        
        # 後驗狀態估計
        self.posteri_estimate = 0.0
        # 後驗誤差協方差
        self.posteri_error_covariance = 1.0

    def update(self, measurement):
        # 預測階段
        priori_estimate = self.posteri_estimate
        priori_error_covariance = self.posteri_error_covariance + self.process_variance

        # 更新階段
        kalman_gain = priori_error_covariance / (priori_error_covariance + self.measurement_variance)
        self.posteri_estimate = priori_estimate + kalman_gain * (measurement - priori_estimate)
        self.posteri_error_covariance = (1 - kalman_gain) * priori_error_covariance

        return self.posteri_estimate



class MV_on_Vechicle:
    def __init__(self):
        # 讀MV內參
        with open("mvs.yaml", "r") as file:
            mvs_data = yaml.load(file, Loader=yaml.FullLoader)
            self.cameraMatrix = np.array(mvs_data['camera_matrix']['data'])
            self.cameraMatrix = self.cameraMatrix.reshape(3,3)
            self.distortion_coefficients = np.array(mvs_data['distortion_coefficients']['data'])
            self.distortion_coefficients = self.distortion_coefficients.reshape(1,5)
        self.lr_window_number = 40
        self.threshold = 8000
        self.frame_width = 640
        self.frame_height = 480


		# specify directory and file name
        # self.dir_path = "mvs_mp4\\0521"
        self.dir_path = "/media/mvclab/HDD/mvs_mp4/0701/gray"  # mvclab
        # self.all_file = os.listdir(self.dir_path)
        # self.all_file = sorted(self.all_file)
        # self.all_file = ["test_2024-03-18-07-57-26_mvs_compressed.mp4"] # 0318
        # self.all_file = ["test_2024-05-21-08-08-41_mvs_compressed.mp4"] # 0521
        self.all_file = ["test_2024-07-01-02-35-07_mvs_compressed.mp4"] # 0701
        # self.all_file = ["test_2024-06-28-10-11-20_mvs.mp4"]

  
        # set parameters for text drawn on the frames
        self.font = cv.FONT_HERSHEY_COMPLEX
        self.fontScale = 1
        self.fontColor = (68, 148, 213)
        self.lineType  = 3

        # 左右轉數值邊界
        self.leaning_right = 24
        self.leaning_left = 16

        self.lr_window_list = [] # 存每個window(左右)
        self.lr_window_state = [] # 每個window的區域結果
        self.polygon_list = [] # 畫每個window的範圍
        self.lr_window_result = [] # 存每個window的結果
        self.record = []
        self.lr_last_state = [] # 方向點數量沒超過閥值的話就使用上一次的結果
        self.lr_center_list = [] # 紀錄中央點，畫圖用
        self.lr_center_without_avg_list = []

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
    
    def is_overtake(self, img, x1, y1, x2, y2):
        det_center = (x1+x2) // 2
        box = img[int(y1):int(y2), int(x1):int(x2)]
        threshold = box.shape[0]*box.shape[1] // 10

        translation = np.ravel(box) # 把img變為一維
        right_index = np.where(translation < 128)
        left_index = np.where(translation > 128)
        diff = len(right_index[0]) - len(left_index[0])

        if det_center < self.frame_width // 3:  # 框框在整個畫面的左邊
            if diff > threshold:    # 框框在左邊，而且框框內的mv值往右 -> 框框內的車是超車
                return True
            else:
                return False
        elif det_center > self.frame_width // 3 * 2:    # 框框在整個畫面的右邊
            if diff < -1 * threshold:    # 框框在右邊，而且框框內的mv值往左
                return True
            else:
                return False


    def run_split_window(self):
        for file in self.all_file:
            filename = file


            # initialise stream from video
            cap = cv.VideoCapture(os.path.join(self.dir_path, filename))
            # cap = cv.VideoCapture(filename)

            print(os.path.join(self.dir_path, filename))
            print(cap.isOpened())
            ret, prvs = cap.read()

            # initialise video writer
            frameRate = int(cap.get(cv.CAP_PROP_FPS))
            codec = cv.VideoWriter_fourcc(*'mp4v')
            save_name = "motion_" + filename[:-4] + ".mp4"
            outputStream = cv.VideoWriter(save_name, codec, frameRate, (int(cap.get(3)),int(cap.get(4))))



            # initialise text variables to draw on frames
            # motion_list = []
            # realMotion = 'None'


            frame_id = 0
            self.frame_width = int(cap.get(3))
            self.frame_height = int(cap.get(4))
            window_width = self.frame_width // self.lr_window_number

            # 前面的frame_height-(frame_height//10 * 2)是為了不要底下雨刷的部份，不要底下的1/5範圍
            # window_height = (frame_height-int(frame_height//2)) // self.ud_window_number
            window_left = self.frame_width // 4
            window_right = self.frame_width // 4 * 3
            # window_left = 0
            # window_right = frame_width

            window_bottom = self.frame_height // 4 * 3
            window_top = self.frame_height // 3


            for i in range(self.lr_window_number):
                self.lr_last_state.append("")
                self.lr_window_result.append([])
                
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




                yuv_with_polygons = nxt.copy()

                # 如果目前在轉彎，則不要進行物件偵測輔助
                if len(self.lr_center_list) > 20:
                    if self.lr_center_list[-1] < self.lr_window_number//4 or self.lr_center_list[-1] > self.lr_window_number//4*3:
                        cv.circle(yuv_with_polygons, (10,10), 10, (0, 0, 255), -1)
                        self.is_detect = False
                    else:
                        cv.circle(yuv_with_polygons, (10,10), 10, (0, 255, 0), -1)
                        self.is_detect = True
                self.is_detect = True

                # yolo 偵測
                yoloPicture = cv.merge((y, y, y))
                yoloResult = self.model(yoloPicture, verbose=False)
                for result in yoloResult:
                    for box in result.boxes:
                        cls = box.cls
                        classID = cls.item()
                        if classID == 2  or classID==3 or classID == 0 or classID == 7:    # 2:car; 3:motorcycle; 5:bus; 7:truck
                            x1, y1, x2, y2 = box.xyxy[0]
                            if self.is_detect:
                                if self.is_overtake(v, x1, y1, x2, y2):
                                    v[int(y1):int(y2), int(x1):int(x2)] = 128   # 偵測框內的MV值不計算 (128是沒有向量)
                                    conf = box.conf
                                    cv.rectangle(yoloPicture, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                    cv.putText(yoloPicture, f'{self.model.names[int(cls.item())]} {conf.item():.2f}', (int(x1), int(y1)-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                else:
                                    conf = box.conf
                                    cv.rectangle(yoloPicture, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                    cv.putText(yoloPicture, f'{self.model.names[int(cls.item())]} {conf.item():.2f}', (int(x1), int(y1)-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            else:
                                conf = box.conf
                                cv.rectangle(yoloPicture, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv.putText(yoloPicture, f'{self.model.names[int(cls.item())]} {conf.item():.2f}', (int(x1), int(y1)-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv.imshow("yolo", yoloPicture)

                # cv.imwrite("yolo.jpg", yoloPicture)

                self.lr_window_list.clear()
                self.polygon_list.clear()
                self.lr_window_state.clear()

                # 直切
                for i in range(self.lr_window_number):
                    self.lr_window_list.append(v[window_top:window_bottom, window_width*i:window_width*(i+1)])

                    # # 實際偵測範圍
                    # polygon = [[window_width*i, window_top], [window_width*(i+1), window_top], [window_width*(i+1),window_bottom], [window_width*i, window_bottom]]
                    # polygon = np.array([polygon], dtype=np.int32)
                    # self.polygon_list.append(polygon)

                    # 示意框
                    polygon = [[window_width*i, window_top-40], [window_width*(i+1), window_top-40], [window_width*(i+1),window_top -20], [window_width*i, window_top - 20]]
                    polygon = np.array([polygon], dtype=np.int32)
                    self.polygon_list.append(polygon)


                # 畫偵測區域(漸層)
                for i in range(self.lr_window_number):
                    # Calculate blue channel value for gradient
                    blue_value = int(255 * (self.lr_window_number - i) / self.lr_window_number + 80)
                    red_value = int(255 * i / self.lr_window_number + 80)
                    # Draw polygon with calculated color
                    color_bgr = (red_value, 30, blue_value)
                    yuv_with_polygons = cv.polylines(yuv_with_polygons, self.polygon_list[i], isClosed=True, color=color_bgr, thickness=2)


                # 儲存每個window的區域結果(左右)
                for i in range(self.lr_window_number):
                    tempAns_R = self.lr_estimate(self.lr_window_list[i])
                    if(tempAns_R == "None"):
                        self.lr_window_state.append(self.lr_last_state[i])
                    if(tempAns_R != "None"):
                        self.lr_last_state[i] = tempAns_R
                        self.lr_window_state.append(tempAns_R)
                    

                # 畫上每個區域結果(左右)
                lr_tempRow = []
                for i in range(self.lr_window_number):
                    # cv.putText(yuv_with_polygons, str(window_state[i]), (window_width*i+20, 100), self.font, self.fontScale, self.fontColor, self.lineType)
                    if(self.lr_window_state[i] == ""):
                        self.lr_window_result[i].append(0)
                        lr_tempRow.append(0)
                    else:
                        self.lr_window_result[i].append(int(self.lr_window_state[i]))
                        lr_tempRow.append(int(self.lr_window_state[i]))
                


                lr_center = self.find_center(lr_tempRow)
                self.lr_center_without_avg_list.append(lr_center)


                # 20 幀後才開始算
                if len(self.lr_center_without_avg_list) >= 20:
                    lr_center_sum = 0

                    for i in range(1, 21):
                        lr_center_sum += self.lr_center_without_avg_list[-i]

                    lr_center_avg = int(lr_center_sum / 20)

                    # 畫中心位置
                    self.lr_center_list.append(lr_center_avg)
                    cv.circle(yuv_with_polygons, ((self.frame_width*lr_center_avg//self.lr_window_number)+(window_width//2), window_top-30), 6, (31, 198, 0), -1)


                    # if center_avg == 0:
                    #     motion_list.append("Left")
                    # elif center_avg == self.lr_window_number - 1:
                    #     motion_list.append("Right")
                    # elif center_avg >= self.leaning_left and center_avg <= self.leaning_right:
                    #     motion_list.append("Straight")
                    # elif center_avg < self.leaning_left:
                    #     motion_list.append("Leaning left")
                    # elif center_avg > self.leaning_right:
                    #     motion_list.append("Leaning right")
                else:
                    self.lr_center_list.append(0) # 為了畫圖合理，前面20幀沒有數據補0


                    
                # if(len(motion_list) >= 3):
                #     if(motion_list[0] == motion_list[1] and motion_list[1] == motion_list[2]):
                #         realMotion = motion_list[0]


                # cv.putText(yuv_with_polygons, str(realMotion), (260,400), self.font, self.fontScale, (0, 0, 255), self.lineType)
                cv.putText(yuv_with_polygons, str(frame_id), (610, 20), self.font, 0.5, (0, 0, 255), 1)

                # y = cv.cvtColor(y, cv.COLOR_GRAY2BGR)
                # u = cv.cvtColor(u, cv.COLOR_GRAY2BGR)
                # v = cv.cvtColor(v, cv.COLOR_GRAY2BGR)

                # for result in yoloResult:
                #     for box in result.boxes:
                #         cls = box.cls
                #         classID = cls.item()
                #         if classID == 2  or classID==3 or classID == 0 or classID == 7:    # 2:car; 3:motorcycle; 5:bus; 7:truck
                #             x1, y1, x2, y2 = box.xyxy[0]
                #             # v[int(y1):int(y2), int(x1):int(x2)] = 128   # 偵測框內的MV值不計算 (128是沒有向量)
                #             conf = box.conf
                #             cv.rectangle(yuv_with_polygons, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                #             cv.putText(yuv_with_polygons, f'{self.model.names[int(cls.item())]} {conf.item():.2f}', (int(x1), int(y1)-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                cv.imshow("polygons", yuv_with_polygons)

                # outputV = cv.merge((y,y,y))
                # cv.imshow("gray", y)
                # cv.imshow("y", y)
                # cv.imshow("u", u)
                cv.imshow("v", v)
                cv.imwrite("gray.jpg", y)
                cv.imwrite("v.jpg", v)
                # cv.imwrite("polygons.jpg", yuv_with_polygons)
                # for i in range(self.window_number):
                #     cv.imshow(str(i), self.lr_window_list[i])


                if cv.waitKey(25) & 0xFF == ord('q'):
                    break
                # 不用 YOLO 偵測的話會跑太快
                # if cv.waitKey(1000//frameRate) & 0xFF == ord('q'):
                #     break
                
                outputV = cv.merge((v,v,v))
                outputStream.write(yoloPicture)

                frame_id += 1


            outputStream.release()



if __name__ == "__main__":

    detector = MV_on_Vechicle()
    detector.run_split_window()
    # plt.cla()

    # plt.plot()
    # plt.plot(detector.lr_center_list, label="mapped")

    # plt.xlabel("Frame", {'fontsize':20})
    # plt.ylabel("Center", {'fontsize':20})
    # plt.legend(
    #     loc='best',
    #     fontsize=20,
    #     shadow=True,
    #     facecolor='#ccc',
    #     edgecolor='#000',
    #     title='test',
    #     title_fontsize=20)
    # plt.savefig("plot.png")
    # plt.show()
    # MV_on_Vechicle().run_split_window()
    # MV_on_Vechicle().run_two_window()
