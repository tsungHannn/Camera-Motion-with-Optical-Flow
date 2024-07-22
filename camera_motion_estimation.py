import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import yaml
from ultralytics import YOLO
"""
MVS 資料格式：
輸出為640*480 YUV422，其中
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
        self.lr_window_number = 40
        self.ud_window_number = 10
        self.threshold = 8000

		# specify directory and file name
        # self.dir_path = "mvs_mp4\\0521"
        self.dir_path = "/media/mvclab/HDD/mvs_mp4/0701/gray"  # mvclab
        # self.all_file = os.listdir(self.dir_path)
        # self.all_file = sorted(self.all_file)
        # self.all_file = ["test_2024-03-18-07-57-26_mvs_compressed.mp4"] # 0318
        # self.all_file = ["test_2024-05-21-08-08-41_mvs_compressed.mp4"] # 0521
        self.all_file = ["test_2024-07-01-02-28-03_mvs_compressed.mp4"] # 0701
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
        self.ud_window_list_R = [] # 上下window
        self.ud_window_list_L = []
        self.lr_window_state = [] # 每個window的區域結果
        self.ud_window_state_R = []
        self.ud_window_state_L = []
        self.polygon_list = [] # 畫每個window的範圍
        self.lr_window_result = [] # 存每個window的結果
        self.ud_window_result_R = []
        self.ud_window_result_L = []
        self.record = []
        self.lr_last_state = [] # 方向點數量沒超過閥值的話就使用上一次的結果
        self.ud_last_state_R = []
        self.ud_last_state_L = []
        self.lr_center_list = [] # 紀錄中央點，畫圖用
        self.ud_center_list_R = []
        self.ud_center_list_L = []
        self.lr_center_without_avg_list = []
        self.ud_center_without_avg_list_R = []
        self.ud_center_without_avg_list_L = []



        self.model = YOLO('yolov8m.pt')
        
    # estimate left or right
    def lr_estimate(self, img):
        self.threshold = img.shape[0]*img.shape[1]//3
        
        translation = np.ravel(img)

        right_index = np.where(translation < 128)
        left_index = np.where(translation > 128)

        diff = len(right_index[0]) - len(left_index[0])
        if diff > self.threshold:
            return 1 # 向右
        elif diff < -1 * self.threshold:
            return -1 # 向左
        else:
            return "None"
    
    # # estimate up or down
    # def ud_estimate(self, img):
    #     self.threshold = img.shape[0]*img.shape[1]//10
        
    #     translation = np.ravel(img)

    #     down_index = np.where(translation < 128)
    #     up_index = np.where(translation > 128)

    #     diff = len(down_index[0]) - len(up_index[0])
    #     if diff > self.threshold:
    #         return 1 # 向下
    #     elif diff < -1 * self.threshold:
    #         return -1 # 向上
    #     else:
    #         return "None"


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
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            window_width = frame_width // self.lr_window_number

            # 前面的frame_height-(frame_height//10 * 2)是為了不要底下雨刷的部份，不要底下的1/5範圍
            window_height = (frame_height-int(frame_height//2)) // self.ud_window_number
            window_left = frame_width // 4
            window_right = frame_width // 4 * 3
            # window_left = 0
            # window_right = frame_width

            window_bottom = frame_height // 4 * 3
            window_top = frame_height // 3


            for i in range(self.lr_window_number):
                self.lr_last_state.append("")
                self.lr_window_result.append([])
                
            

            for i in range(self.ud_window_number):
                self.ud_last_state_R.append("")
                self.ud_window_result_R.append([])
                self.ud_last_state_L.append("")
                self.ud_window_result_L.append([])
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


                # yolo 偵測
                yoloPicture = cv.merge((y, y, y))
                yoloResult = self.model(yoloPicture, verbose=False)
                # for result in yoloResult:
                #     for box in result.boxes:
                #         cls = box.cls
                #         classID = cls.item()
                #         if classID == 2  or classID==3 or classID == 0 or classID == 7:    # 2:car; 3:motorcycle; 5:bus; 7:truck
                #             x1, y1, x2, y2 = box.xyxy[0]
                #             # v[int(y1):int(y2), int(x1):int(x2)] = 128   # 偵測框內的MV值不計算 (128是沒有向量)
                #             conf = box.conf
                            # cv.rectangle(yoloPicture, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            # cv.putText(yoloPicture, f'{self.model.names[int(cls.item())]} {conf.item():.2f}', (int(x1), int(y1)-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # cv.imshow("yolo", yoloPicture)
               
                # cv.imwrite("yolo.jpg", yoloPicture)

                self.lr_window_list.clear()
                self.ud_window_list_R.clear()
                self.ud_window_list_L.clear()
                self.polygon_list.clear()
                self.lr_window_state.clear()
                self.ud_window_state_R.clear()
                self.ud_window_state_L.clear()

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

                # # 橫切 (把上面切掉)
                # for i in range(self.ud_window_number):
                #     # 中間
                #     # self.ud_window_list_R.append(u[window_top+window_height*i:window_top+window_height*(i+1), window_left:window_right])
                #     # 分成左右
                #     self.ud_window_list_R.append(u[window_top+window_height*i:window_top+window_height*(i+1), 0:frame_width//3])
                #     self.ud_window_list_L.append(u[window_top+window_height*i:window_top+window_height*(i+1), frame_width//3*2:frame_width])
                #     # # 實際偵測範圍
                #     # 中間
                #     # polygon = [[window_left, window_top+window_height*i], [window_right, window_top+window_height*i], [window_right,window_top+ window_height*(i+1)], [window_left, window_top+window_height*(i+1)]]
                #     # polygon = np.array([polygon], dtype=np.int32)
                #     # self.polygon_list.append(polygon)
                #     # 分成左右
                #     polygon = [[0, window_top+window_height*i], [frame_width//3, window_top+window_height*i], [frame_width//3, window_top+window_height*(i+1)], [0, window_top+window_height*(i+1)]]
                #     polygon = np.array([polygon], dtype=np.int32)
                #     self.polygon_list.append(polygon)

                #     polygon = [[frame_width//3*2, window_top+window_height*i], [frame_width, window_top+window_height*i], [frame_width, window_top+window_height*(i+1)], [frame_width//3*2, window_top+window_height*(i+1)]]
                #     polygon = np.array([polygon], dtype=np.int32)
                #     self.polygon_list.append(polygon)


                #     # # 示意框
                #     # polygon = [[window_right-20, window_top+window_height*i], [window_right, window_top+window_height*i], [window_right, window_top+window_height*(i+1)], [window_right-20, window_top+window_height*(i+1)]]
                #     # polygon = np.array([polygon], dtype=np.int32)
                #     # self.polygon_list.append(polygon)
                
                # # 橫切 (保留最上面)
                # for i in range(self.ud_window_number):
                #     # self.ud_window_list_r.append(u[window_height*i:window_height*(i+1), window_left:window_right])
                #     self.ud_window_list_R.append(u[window_height*i:window_height*(i+1), 0:frame_width//3])
                #     self.ud_window_list_L.append(u[window_height*i:window_height*(i+1), frame_width//3*2:frame_width])


                #     # 實際偵測範圍
                #     # polygon = [[window_left, window_height*i], [window_right, window_height*i], [window_right, window_height*(i+1)], [window_left, window_height*(i+1)]]
                #     # polygon = np.array([polygon], dtype=np.int32)
                #     polygon = [[0, window_height*i], [frame_width//3, window_height*i], [frame_width//3, window_height*(i+1)], [0, window_height*(i+1)]]
                #     polygon = np.array([polygon], dtype=np.int32)
                #     self.polygon_list.append(polygon)

                #     polygon = [[frame_width//3*2, window_height*i], [frame_width, window_height*i], [frame_width, window_height*(i+1)], [frame_width//3*2, window_height*(i+1)]]
                #     polygon = np.array([polygon], dtype=np.int32)
                #     self.polygon_list.append(polygon)
                #     # # 示意框
                #     # polygon = [[window_right-20,window_height*i], [window_right, window_height*i], [window_right, window_height*(i+1)], [window_right-20, window_height*(i+1)]]
                #     # polygon = np.array([polygon], dtype=np.int32)
                #     # self.polygon_list.append(polygon)


                # 畫偵測區域(漸層)
                yuv_with_polygons = nxt.copy()

                for i in range(self.lr_window_number):
                    # Calculate blue channel value for gradient
                    blue_value = int(255 * (self.lr_window_number - i) / self.lr_window_number + 80)
                    red_value = int(255 * i / self.lr_window_number + 80)
                    # Draw polygon with calculated color
                    color_bgr = (red_value, 30, blue_value)
                    yuv_with_polygons = cv.polylines(yuv_with_polygons, self.polygon_list[i], isClosed=True, color=color_bgr, thickness=2)

                # count = 0
                # for i in range(self.lr_window_number, len(self.polygon_list)):
                    
                #     # Calculate blue channel value for gradient
                #     blue_value = int(255 * (self.ud_window_number - count) / self.ud_window_number + 80)
                #     red_value = int(255 * count / self.ud_window_number + 80)
                #     # Draw polygon with calculated color
                #     color_bgr = (red_value, 30, blue_value)
                #     yuv_with_polygons = cv.polylines(yuv_with_polygons, self.polygon_list[i], isClosed=True, color=color_bgr, thickness=2)
                #     count+=1
                    
                    
                

# # 分三種顏色
#                     if i == 0:
#                         yuv_with_polygons = cv.polylines(yuv_with_polygons, self.polygon_list[i], isClosed=True, color=(0, 255, 0), thickness=2)
#                     elif i == self.window_number - 1:
#                         yuv_with_polygons = cv.polylines(yuv_with_polygons, self.polygon_list[i], isClosed=True, color=(0, 255, 0), thickness=2)
#                     elif i >= self.leaning_left and i <= self.leaning_right:
#                         yuv_with_polygons = cv.polylines(yuv_with_polygons, self.polygon_list[i], isClosed=True, color=(255, 0, 0), thickness=2)
#                     elif i < self.leaning_left:
#                        yuv_with_polygons = cv.polylines(yuv_with_polygons, self.polygon_list[i], isClosed=True, color=(0, 255, 255), thickness=2)
#                     elif i > self.leaning_right:
#                         yuv_with_polygons = cv.polylines(yuv_with_polygons, self.polygon_list[i], isClosed=True, color=(0, 255, 255), thickness=2)

                    
                # 儲存每個window的區域結果(左右)
                for i in range(self.lr_window_number):
                    tempAns_R = self.lr_estimate(self.lr_window_list[i])
                    if(tempAns_R == "None"):
                        self.lr_window_state.append(self.lr_last_state[i])
                    if(tempAns_R != "None"):
                        self.lr_last_state[i] = tempAns_R
                        self.lr_window_state.append(tempAns_R)
                    
                
                # # 儲存每個window的區域結果(上下)
                # for i in range(self.ud_window_number):
                #     tempAns_R = self.ud_estimate(self.ud_window_list_R[i])
                #     tempAns_L = self.ud_estimate(self.ud_window_list_L[i])
                #     if(tempAns_R == "None"):
                #         self.ud_window_state_R.append(self.ud_last_state_R[i])
                #     if(tempAns_R != "None"):
                #         self.ud_last_state_R[i] = tempAns_R
                #         self.ud_window_state_R.append(tempAns_R)
                #     if(tempAns_L == "None"):
                #         self.ud_window_state_L.append(self.ud_last_state_L[i])
                #     if(tempAns_L != "None"):
                #         self.ud_last_state_L[i] = tempAns_L
                #         self.ud_window_state_L.append(tempAns_L)


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
                
                # # 畫上每個區域結果(上下)
                # ud_tempRow_R = []
                # ud_tempRow_L = []
                # for i in range(self.ud_window_number):
                #     # cv.putText(yuv_with_polygons, str(self.ud_window_state[i]), (10, window_height*i+30), self.font, self.fontScale, self.fontColor, self.lineType)
                #     if(self.ud_window_state_R[i] == ""):
                #         self.ud_window_result_R[i].append(0)
                #         ud_tempRow_R.append(0)
                #     else:
                #         self.ud_window_result_R[i].append(int(self.ud_window_state_R[i]))
                #         ud_tempRow_R.append(int(self.ud_window_state_R[i]))
                #     if(self.ud_window_state_L[i] == ""):
                #         self.ud_window_result_L[i].append(0)
                #         ud_tempRow_L.append(0)
                #     else:
                #         self.ud_window_result_L[i].append(int(self.ud_window_state_L[i]))
                #         ud_tempRow_L.append(int(self.ud_window_state_L[i]))


                lr_center = self.find_center(lr_tempRow)
                self.lr_center_without_avg_list.append(lr_center)
                # ud_center_R = self.find_center(ud_tempRow_R)
                # self.ud_center_without_avg_list_R.append(ud_center_R)
                # ud_center_L = self.find_center(ud_tempRow_L)

                # ud_center = (ud_center_L + ud_center_R) / 2
                # self.ud_center_without_avg_list_L.append(ud_center) # 方便起見就用這個當作左右相加後的平均值


                # 20 幀後才開始算
                if len(self.lr_center_without_avg_list) >= 20:
                    lr_center_sum = 0
                    # ud_center_sum_R = 0
                    # ud_center_sum_L = 0
                    for i in range(1, 21):
                        lr_center_sum += self.lr_center_without_avg_list[-i]
                        # ud_center_sum_R += self.ud_center_without_avg_list_R[-i]
                        # ud_center_sum_L += self.ud_center_without_avg_list_L[-i]

                    lr_center_avg = int(lr_center_sum / 20)
                    # ud_center_avg_R = int(ud_center_sum_R / 20)
                    # ud_center_avg_L = int(ud_center_sum_L / 20)

                    # 畫中心位置
                    self.lr_center_list.append(lr_center_avg)
                    cv.circle(yuv_with_polygons, ((frame_width*lr_center_avg//self.lr_window_number)+(window_width//2), window_top-30), 6, (31, 198, 0), -1)
                    
                    # self.ud_center_list_R.append(ud_center_avg_R)
                    # self.ud_center_list_L.append(ud_center_avg_L)
                    # # 把上面切掉
                    # cv.circle(yuv_with_polygons, (window_right-10, window_top+window_height*ud_center_avg_R + 15), 6, (255, 0, 0), -1)
                    # cv.circle(yuv_with_polygons, (frame_width//2, window_top+window_height*ud_center_avg_L + 15), 6, (0, 0, 255), -1)

                    # # 保留最上面
                    # cv.circle(yuv_with_polygons, (window_right-10, window_height*ud_center_avg_R + 15), 6, (255, 0, 0), -1)
                    # cv.circle(yuv_with_polygons, (10, window_height*ud_center_avg_L + 15), 6, (255, 0, 0), -1)
                    # cv.circle(yuv_with_polygons, (frame_width-10, window_height*ud_center_avg_R + 15), 6, (255, 0, 0), -1)
                    # cv.circle(yuv_with_polygons, (frame_width//2, window_height*ud_center_avg_L + 15), 6, (0, 0, 255), -1)
                    # if len(motion_list) >= 3:
                    #     motion_list.pop(0)


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
                # cv.putText(yuv_with_polygons, str(frame_id), (0, 20), self.font, 0.5, (0, 0, 255), 1)

                y = cv.cvtColor(y, cv.COLOR_GRAY2BGR)
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

                # cv.imshow("y", y)
                # cv.imshow("u", u)
                cv.imshow("v", v)
                # cv.imwrite("polygons.jpg", yuv_with_polygons)
                # for i in range(self.window_number):
                #     cv.imshow(str(i), self.lr_window_list[i])

                # if cv.waitKey(25) & 0xFF == ord('q'):
                #     break
                
                outputV = cv.merge((v,v,v))
                outputStream.write(outputV)

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
