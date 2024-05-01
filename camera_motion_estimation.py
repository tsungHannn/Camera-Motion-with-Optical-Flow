import os
import time
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
import yaml

class MV_on_Vechicle:
    def __init__(self):
        with open("mvs.yaml", "r") as file:
            mvs_data = yaml.load(file)
            self.cameraMatrix = np.array(mvs_data['camera_matrix']['data'])
            self.cameraMatrix = self.cameraMatrix.reshape(3,3)
            self.distortion_coefficients = np.array(mvs_data['distortion_coefficients']['data'])
            self.distortion_coefficients = self.distortion_coefficients.reshape(1,5)
        self.scene = "outside"
        self.window_number = 40
        self.threshold = 8000

		# specify directory and file name
        # self.dir_path = "mvs_mp4\\0318"
        self.dir_path = "/media/mvclab/HDD/mvs_mp4/0318"  # mvclab

		# all_file = os.listdir(dir_path)
		# all_file = [os.path.join(dir_path, "0419_2024-04-19-06-08-40_mvs_compressed.mp4")]
        self.all_file = ["test_2024-03-18-07-57-26_mvs_compressed.mp4"]
		# all_file = ["0419_2024-04-19-06-08-40_mvs_compressed.mp4"]
		# all_file = ["test_2024-03-18-08-00-02_mvs_compressed.mp4"] # 戶外直線+右轉(畫數據圖的)
		# all_file = [os.path.join(dir_path, "test_2024-03-18-07-59-24_mvs_compressed.mp4")]


		# filename = "/media/mvclab/HDD/mvs_mp4/test_2024-03-18-08-00-56_mvs_compressed.mp4" # mvclab 戶外直線
		# filename = "test_2024-03-18-08-00-56_mvs_compressed.mp4" # 戶外直線
		# filename = "test_2024-03-18-07-59-24_mvs_compressed.mp4" # 戶外直線+右轉
		# filename = "test_2024-03-18-07-57-26_mvs_compressed.mp4" # 最長的
		# filename = "test_2024-03-18-08-05-15_mvs_compressed.mp4" # 室內

		# filename = "T2_301_01.mp4"
  
        # set parameters for text drawn on the frames
        self.font = cv.FONT_HERSHEY_COMPLEX
        self.fontScale = 1
        self.fontColor = (68, 148, 213)
        self.lineType  = 3
        # 儲存每幀
        self.save_frame = False
        self.save = "save"
        # 左右轉數值邊界
        self.leaning_right = 24
        self.leaning_left = 16

        self.window_list = [] # 存每個window
        self.window_state = [] # 每個window的區域結果
        self.polygon_list = [] # 畫每個window的範圍
        self.window_result = [] # 存每個window的結果
        self.record = []
        self.last_state = []
        self.center_list = []
        self.center_without_avg_list = []
        

    # 輸入影像 輸出目前影像是往左還是右
    def yuv_estimate(self, img):
        self.threshold = img.shape[0]*img.shape[1]//3
        # print("threshold:", self.threshold)
        
        translation = np.ravel(img)
        # nonzeros = np.where(translation != 128)
        # translation = translation[nonzeros]
        left_index = np.where(translation < 128)
        right_index = np.where(translation > 128)

        left = translation[left_index]
        right = translation[right_index]

        # value = 0
        # if len(left_index[0]) < 5000 and len(right_index[0]) < 5000:
        #     return "stop"
        if self.scene == "outside":
            diff = len(left_index[0]) - len(right_index[0])
            # cv.putText(img, str(diff), (50,100), font, fontScale, fontColor, lineType)
            if diff > self.threshold:
                # return int((np.sum(left)/left.size) - 128)
                return 1
                # value = mode(left)[0][0]
                return "left"
            elif diff < -1 * self.threshold:
                # return int((np.sum(right)/right.size) - 128)
                return -1
                # value = mode(right)[0][0]
                return "right"
            else:
                return "None"
            # if len(left_index[0]) > len(right_index[0]):
            #     value = mode(left)[0][0]
            # elif len(left_index[0]) < len(right_index[0]):
            #     value = mode(right)[0][0]
        elif self.scene == "inside":
            diff = len(left_index[0]) - len(right_index[0])
            cv.putText(img, str(diff), (50,100), self.font, self.fontScale, self.fontColor, self.lineType)
            if diff > 2000:
                # value = mode(left)[0][0]
                return "left"
            elif diff < -2000:
                # value = mode(right)[0][0]
                return "right"
            else:
                return "None"
            # if len(left_index[0]) > len(right_index[0]):
            #     value = mode(left)[0][0]
            # elif len(left_index[0]) < len(right_index[0]):
            #     value = mode(right)[0][0]


    # 把MV得到的X,Y軸向量加起來，得到每個pixel的大小跟角度
    def getVanishingPoint(self, u, v):
        magnitude = np.sqrt(u**2 + v**2)
        angle = np.arctan2(u, v)

        # 將角度轉換為 0 到 360 的範圍
        angle = ((angle + np.pi) * 180 / np.pi) % 360
        # 繪製合併的位移向量場
        u = u.astype("int16")
        u = u - 128
        v = v.astype("int16")
        v = v - 128


        plt.quiver(u, v, color="red", scale=0.01)  # scale 控制箭頭的大小
        plt.title('Combined Displacement Vector Field')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()



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
                best_index = self.window_number - 1
            elif np.sum(arr) > 0:
                best_index = 0

        return best_index
    
    def exponential_moving_average(self, arr, alpha=0.5):
        ema = arr[0]  # 初始 EMA 設為第一個元素
        for i in range(1, len(arr)):
            ema = alpha * arr[i] + (1 - alpha) * ema
        return ema


    def run(self):
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
            angle = 'None'
            translation = 'None'
            motion = 'None'
            motion_type = 'None'
            motion_list = []
            motion_index = -1
            realMotion = 'None'
            # set counter value
            # count = 1



            if self.scene == "outside":
                frame_id = 0
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))
                window_width = frame_width // self.window_number

                window_bottom = frame_height // 4 * 3
                window_top = frame_height // 8
                # window_bottom = frame_height // 7*6
                # window_top = frame_height // 7

                left_width = frame_width // 3
                right_wide = left_width * 2
                
            elif self.scene == "inside":
                frame_id = 0
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))
                left_width = frame_width // 3
                right_wide = left_width * 2
                window_top = frame_height // 10
                window_bottom = frame_height // 5 * 4

            noiseList1 = []
            noiseList2 = []
            averageList = []
            last_left = ""
            last_right = ""


            for i in range(self.window_number):
                self.last_state.append("")
            
            for i in range(self.window_number):
                self.window_result.append([])
            # main loop
            while True:
                # read a new frame
                ret, nxt = cap.read()

                if not ret:
                    break
                
                cv.imshow("before", nxt)
                nxt = cv.undistort(nxt, cameraMatrix=self.cameraMatrix, distCoeffs=self.distortion_coefficients)
                # cv.imshow("after", nxt)

                yuv = cv.cvtColor(nxt.copy(), cv.COLOR_RGB2YUV)

                y, u, v = cv.split(yuv)


                # getVanishingPoint(u.copy(), v.copy())

                self.window_list.clear()
                self.polygon_list.clear()
                self.window_state.clear()
                for i in range(self.window_number):
                    self.window_list.append(v[window_top:window_bottom, window_width*i:window_width*(i+1)])
                    polygon = [[window_width*i, window_top], [window_width*(i+1), window_top], [window_width*(i+1),window_bottom], [window_width*i, window_bottom]]
                    polygon = np.array([polygon], dtype=np.int32)
                    self.polygon_list.append(polygon)


                

                left_img = v[window_top:window_bottom, :left_width]
                right_img = v[window_top:window_bottom, right_wide:]
                # left_polygon_pts = [[0, window_top], [left_width - 1, window_top], [left_width - 1, window_bottom - 1], [0, window_bottom - 1]]
                # right_polygon_pts = [[right_wide, window_top], [frame_width - 1, window_top], [frame_width - 1, window_bottom - 1], [right_wide, window_bottom - 1]]
                # left_polygon_pts = np.array([left_polygon_pts], dtype=np.int32)
                # right_polygon_pts = np.array([right_polygon_pts], dtype=np.int32)
                yuv_with_polygons = nxt.copy()
                for i in range(self.window_number):
                    if i == 0:
                        yuv_with_polygons = cv.polylines(yuv_with_polygons, self.polygon_list[i], isClosed=True, color=(0, 255, 0), thickness=2)
                    elif i == self.window_number - 1:
                        yuv_with_polygons = cv.polylines(yuv_with_polygons, self.polygon_list[i], isClosed=True, color=(0, 255, 0), thickness=2)
                    elif i >= self.leaning_left and i <= self.leaning_right:
                        yuv_with_polygons = cv.polylines(yuv_with_polygons, self.polygon_list[i], isClosed=True, color=(255, 0, 0), thickness=2)
                    elif i < self.leaning_left:
                       yuv_with_polygons = cv.polylines(yuv_with_polygons, self.polygon_list[i], isClosed=True, color=(0, 255, 255), thickness=2)
                    elif i > self.leaning_right:
                        yuv_with_polygons = cv.polylines(yuv_with_polygons, self.polygon_list[i], isClosed=True, color=(0, 255, 255), thickness=2)
                    
                    
                # yuv_with_polygons = cv.polylines(nxt.copy(), left_polygon_pts, isClosed=True, color=(0, 255, 0), thickness=2)
                # yuv_with_polygons = cv.polylines(yuv_with_polygons, right_polygon_pts, isClosed=True, color=(0, 255, 0), thickness=2)

                # if frame_id % 3 == 0:

                # 用垂直向量檢測雜點（未完成）
                # ===============================================================
                # vertical = np.ravel(u)
                # up = np.where(vertical > 130)
                # down = np.where(vertical < 126)
                # ver_s = (up[0].size + down[0].size) / vertical.size


                # up = np.where(u > 128)
                # down = np.where(u < 128)
                # ver_s = np.sum(u[up]) + np.sum(u[down])
                # ver_s /= (up[0].size + down[0].size)
                
                # left_horizontal = np.ravel(left_img)
                # # right_horizontal = np.ravel(right_img)

                # right1 = np.where(left_horizontal > 128)
                # left1 = np.where(left_horizontal < 128)
                # hor_s1 = abs(right1[0].size - left1[0].size) / (right1[0].size + left1[0].size)

                # right2 = np.where(right_horizontal > 128)
                # left2 = np.where(right_horizontal < 128)
                # hor_s2 = abs(right2[0].size - left2[0].size) / (right2[0].size + left2[0].size)

                # peaks = find_peaks(ver_s)
                # noiseList1.append(hor_s1)
                # average = np.average(noiseList1)
                # averageList.append(average)
                # noiseList2.append(hor_s2)
                # peaks, _ = find_peaks(noiseList, prominence=25000)

                # plotPeak = np.array(noiseList)[peaks]
                # plt.clf()
                # plt.plot(noiseList1, label="left")
                # plt.plot(averageList, label="average")
                # plt.legend()
                # plt.plot(peaks, np.array(noiseList)[peaks], "x")
                # plt.plot(record[0])
                # plt.plot(record[4])
                # plt.pause(0.00001)
                # plt.ioff()
                
                # ===============================================================

                # 儲存每個window的區域結果
                for i in range(self.window_number):
                    tempAns = self.yuv_estimate(self.window_list[i])
                    if(tempAns == "None" and self.last_state != ""):
                        self.window_state.append(self.last_state[i])
                    if(tempAns != "None"):
                        self.last_state[i] = tempAns
                        self.window_state.append(tempAns)
                
                



                # left_state = str(yuv_estimate(left_img))
                # right_state = str(yuv_estimate(right_img))

                # if(last_left != ""):
                #     if(left_state == 'None'):
                #         left_state = last_left
                # if(last_right != ""):
                #     if(right_state == "None"):
                #         right_state = last_right

                # if(left_state != "None"):
                #     last_left = left_state
                # if(right_state != "None"):
                #     last_right = right_state


                # # if left_state == "None" or right_state == "None":
                # #     pass
                # # else:
                    
                # if(left_state == "left") and (right_state == "right"):
                #     if len(motion_list) >= 3:
                #         motion_list.pop(0)
                #     motion_list.append("Straight")
                # elif(left_state == "left") and (right_state == "left"):
                #     if len(motion_list) >= 3:
                #         motion_list.pop(0)
                #     motion_list.append("Right")
                # elif(left_state == "right") and (right_state == "right"):
                #     if len(motion_list) >= 3:
                #         motion_list.pop(0)
                #     motion_list.append("Left")
                # elif(left_state == "stop") and (right_state == "stop"):
                #     if len(motion_list) >= 3:
                #         motion_list.pop(0)
                #     motion_list.append("Stop")
                # else:
                #     pass # stop and right

                # if(len(motion_list) >= 3):
                #     if(motion_list[motion_index - 2] == motion_list[motion_index - 1] and motion_list[motion_index - 1] == motion_list[motion_index]):
                #         realMotion = motion_list[motion_index - 2]

                # 畫上每個區域結果
                tempRow = []
                for i in range(self.window_number):
                    # cv.putText(yuv_with_polygons, str(window_state[i]), (window_width*i+20, 100), self.font, self.fontScale, self.fontColor, self.lineType)
                    if(self.window_state[i] == ""):
                        self.window_result[i].append(0)
                        tempRow.append(0)
                    else:
                        self.window_result[i].append(int(self.window_state[i]))
                        tempRow.append(int(self.window_state[i]))
                

                center = self.find_center(tempRow)
                self.center_without_avg_list.append(center)


                if len(self.center_without_avg_list) >= 20:
                    center_sum = 0
                    # EMA_window = []
                    for i in range(1, 21):
                        center_sum += self.center_without_avg_list[-i]
                        # EMA_window.append(self.center_without_avg_list[-i])
                    

                    # center_avg = int(self.exponential_moving_average(EMA_window))
                    center_avg = int(center_sum / 20)

                    self.center_list.append(center_avg)
                    cv.circle(yuv_with_polygons, (frame_width*center_avg//self.window_number, 30), 5, (0, 0, 255), 2)

                    if len(motion_list) >= 3:
                        motion_list.pop(0)


                    if center_avg == 0:
                        motion_list.append("Left")
                    elif center_avg == self.window_number - 1:
                        motion_list.append("Right")
                    elif center_avg >= self.leaning_left and center_avg <= self.leaning_right:
                        motion_list.append("Straight")
                    elif center_avg < self.leaning_left:
                        motion_list.append("Leaning left")
                    elif center_avg > self.leaning_right:
                        motion_list.append("Leaning right")


                if(len(motion_list) >= 3):
                    if(motion_list[motion_index - 2] == motion_list[motion_index - 1] and motion_list[motion_index - 1] == motion_list[motion_index]):
                        realMotion = motion_list[motion_index - 2]




                # cv.putText(yuv_with_polygons, left_state, (50,100), font, fontScale, fontColor, lineType)
                # cv.putText(yuv_with_polygons, right_state, (450,100), font, fontScale, fontColor, lineType)
                # cv.putText(yuv_with_polygons, str(motion_list), (50,200), font, fontScale, fontColor, lineType)
                cv.putText(yuv_with_polygons, str(realMotion), (260,400), self.font, self.fontScale, (0, 0, 255), self.lineType)
                

                y = cv.cvtColor(y, cv.COLOR_GRAY2BGR)
                u = cv.cvtColor(u, cv.COLOR_GRAY2BGR)
                v = cv.cvtColor(v, cv.COLOR_GRAY2BGR)
                # cv.imshow("Original", nxt)
                # cv.imshow("yuv", yuv)
                # cv.imshow('YImage', y)
                # cv.imshow('UImage', u)
                # cv.imshow('VImage', v)
                # cv.imshow("left", left_img)
                # cv.imshow("right", right_img)
                cv.imshow("polygons", yuv_with_polygons)

                if self.save_frame:
                    cv.imwrite(self.save + "\\" + str(frame_id) + ".jpg", yuv_with_polygons)

                if cv.waitKey(25) & 0xFF == ord('q'):
                    break
                
                outputStream.write(yuv_with_polygons)
                

                frame_id += 1


            # plt.plot(noiseList1, label="left")
            # plt.plot(averageList, label="average")
            # record = np.array(record)
            plt.plot(self.center_list, label="avg")
            plt.plot(self.center_without_avg_list, label="without_avg")
            # plt.plot(record[0])
            # plt.plot(record[2])
            plt.legend(
                loc='best',
                fontsize=20,
                shadow=True,
                facecolor='#ccc',
                edgecolor='#000',
                title='test',
                title_fontsize=20)
            plt.savefig("plot.png")
            plt.show()

            outputStream.release()



if __name__ == "__main__":
    MV_on_Vechicle().run()
