import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2 as cv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import yaml
from utils import KalmanFilter
from lane_detection import lane_detection, display_lines, find_vanishing_point_by_lane
import math
import collections
from optical_flow_find_vp import *

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
                    

        
        self.window_number = 40
        self.threshold = 8000
        self.frame_width = 640
        self.frame_height = 480


		# specify directory and file name
        # self.dir_path = "mvs_mp4\\1108\\edge"
        self.dir_path = "/media/mvclab/HDD/mvs_mp4/1108/gray"  # mvclab
        # self.all_file = os.listdir(self.dir_path)
        # self.all_file = sorted(self.all_file)
        # print("all_file:", self.all_file)
        # self.all_file = ["test_2024-03-18-07-57-26_mvs_compressed.mp4"] # 0318
        # self.all_file = ["test_2024-05-21-08-08-41_mvs_compressed.mp4"] # 0521
        # self.all_file = ["test_2024-07-01-02-38-53_mvs_compressed.mp4"] # 0701 edge
        # self.all_file = ["test_2024-07-01-02-33-02_mvs_compressed.mp4"] # 0701 gray
        self.all_file = ["2024-11-08-05-16-48_mvs_compressed.mp4"] # 1108 edge
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
        self.comp_last_state = []
        self.center_list = [] # 紀錄中央點，畫圖用
        self.comp_center_list = []
        self.center_without_avg_list = []
        self.comp_center_without_avg_list = []



        # 設定一個歷史記錄隊列來存過去幾幀的消失點
        history_size = 5  # 記錄過去 5 幀的結果
        self.vp_history = collections.deque(maxlen=history_size)



        for i in range(self.window_number):
            self.last_state.append("")
            self.comp_last_state.append("")

        
        self.last_center = 20
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

                down_sample_size = 2
                down_sample_yuv = yuv.copy()
                down_sample_yuv = cv.resize(down_sample_yuv, (self.frame_width // down_sample_size, self.frame_height // down_sample_size), interpolation=cv.INTER_CUBIC)
                y, u, v = cv.split(yuv) # 不知道為啥 v看起來才是水平向量    

                down_sample_y, down_sample_u, down_sample_v = cv.split(down_sample_yuv)

                # 車道線偵測
                lane_image, gray_with_line, average_line, vanishing_point = lane_detection(cv.merge((y, y, y)))
                # cv.imshow("lane_detection", lane_image)
                cv.imshow("gray_with_line", gray_with_line)


                # 生成光流圖
                dense_flow  = create_dense_optical_flow(down_sample_v, down_sample_u)
                # filter_dense_flow = self.filter_optical_flow_ransac(dense_flow, sampling_step=10)
                

                # test_flow = np.array([
                #         [[0, 0], [0, 0], [0, 0]],
                #         [[0, 0], [0, 0], [0, 0]],
                #         [[-1, 1], [0, 0], [0, 0]]], dtype=np.float32)


                # # 計算經過次數
                # line_count, line_visual = draw_infinite_lines(dense_flow)
    
                # # 正規化數據以便顯示
                # norm_line_count = cv.normalize(line_count, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
                # # 顯示結果
                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                

                # test_vanishing_point, vis, intersections = find_vanishing_point_by_optical_flow(dense_flow)
                # vis_intersection = np.zeros_like(vis)
                # for intersection in intersections:
                #     cv.circle(vis_intersection, tuple(intersection), 2, (0, 0, 255), -1)
                # cv.imshow("Intersection", vis_intersection)
                # 繪製消失點
                # if tuple(test_vanishing_point) != (-1, -1):
                    # test_vanishing_point = self.smooth_vanishing_point(test_vanishing_point[0], test_vanishing_point[1])
                    # cv.circle(vis, tuple(test_vanishing_point), 5, (0, 0, 255), -1)
                # cv.imshow("Vanishing Point", vis)




                # # Visualize optical flow
                # flow_image = visualize_optical_flow(dense_flow)
                # # legend_image = self.generate_flow_legend()
                
                # axes[0].imshow(line_count, cmap='hot', interpolation='nearest')
                # axes[0].set_title('Pixel Line Coverage')
                
                # axes[1].imshow(cv.cvtColor(flow_image, cv.COLOR_BGR2RGB))
                # axes[1].set_title('Flow Image')
                
                # plt.show()



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

                average_v = v_window[3].copy()
                average_v = cv.blur(average_v, (9, 9))



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
                


                # 如果左右點差距小於閥值，就使用上一次的結果
                for i in range(self.window_number):
                    tempAns = self.estimate(self.window_list[i])
                    comp_tempAns = self.estimate(self.comp_window_list[i])

                    
                    if(tempAns != "None"):
                        self.last_state[i] = tempAns
                        self.window_state.append(tempAns)
                    elif(tempAns == "None"):
                        self.window_state.append(self.last_state[i])

                    if(comp_tempAns != "None"):
                        self.comp_last_state[i] = comp_tempAns
                        self.comp_window_state.append(comp_tempAns)

                    elif(comp_tempAns == "None"):
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


                    # 畫中心位置
                    cv.circle(yuv_with_polygons, ((self.frame_width*center_avg//self.window_number)+(window_width//2), window_top-30), 6, (31, 198, 0), -1)
                    # cv.circle(yuv_with_polygons, ((self.frame_width*comp_center_avg//self.window_number)+(window_width//2), window_top-50), 6, (0, 0, 255), -1)
                    

                    

                    

                # else:
                    # self.center_list.append(20) # 為了畫圖合理，前面20幀沒有數據補0
                    # self.comp_center_list.append(20)


                    
                # if(len(motion_list) >= 3):
                #     if(motion_list[0] == motion_list[1] and motion_list[1] == motion_list[2]):
                #         realMotion = motion_list[0]


                # cv.putText(yuv_with_polygons, str(realMotion), (260,400), self.font, self.fontScale, (0, 0, 255), self.lineType)

                cv.putText(yuv_with_polygons, str(frame_id), (590, 20), self.font, 0.5, (0, 0, 255), 1)

                # # 光流偵測消失點
                # vanishing_point_x = tuple(test_vanishing_point)[0]
                # vanishing_point_y = tuple(test_vanishing_point)[1]

                # cv.circle(yuv_with_polygons, (vanishing_point_x * down_sample_size, vanishing_point_y * down_sample_size), 6, (0, 0, 255), -1)


                # 車道線偵測結果畫進MV on Vehicle
                # 車道線偵測結果
                vanishing_point_in_window = math.floor(vanishing_point[0] / (self.frame_width//self.window_number))
                cv.circle(yuv_with_polygons, ((self.frame_width*vanishing_point_in_window//self.window_number)+(window_width//2), window_top-50), 6, (255, 0, 0), -1)
                
                if len(average_line) == 2:  # 確保有兩條車道線
                    vanishing_point = find_vanishing_point_by_lane(average_line[0], average_line[1])
                    if vanishing_point != (-1, -1):
                        cv.circle(yuv_with_polygons, vanishing_point, 6, (255, 0, 0), -1)  # 用藍色標示焦點

                line_image = display_lines(yuv_with_polygons, average_line)
                yuv_with_polygons = cv.addWeighted(yuv_with_polygons, 0.8, line_image, 1, 1)
                

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
                # cv.imshow("Optical Flow", flow_image)
                # cv.imwrite("optical_flow.png", flow_image)
                # cv.imshow("legend", legend_image)
                # cv.imwrite("legend.png", legend_image)
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



                outputV = cv.merge((v,v,v))
                outputResult.write(yuv_with_polygons)
                outputStream1.write(outputV)
                # outputStream2.write(lane_image)
                # outputStream3.write(gray_with_line)


                # if frame_id == 44:
                #     for i in range(40, 60):
                #         for j in range(80, 100):
                #             print(v[i][j], "\t", end="")
                #         print()
                    
                #     print()

                frame_id += 1



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
            # outputStream3.release()
            end_time = time.time()
            execution_time = end_time - start_time
            minutes = int(execution_time // 60)
            seconds = int(execution_time % 60)
            print(f"Execution time: {minutes}min {seconds}sec")
            print("Frame:", frame_id)
            print("Ignored Frame:", ignored_frame)
            # return



if __name__ == "__main__":

    detector = MV_on_Vechicle()
    detector.run()
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
