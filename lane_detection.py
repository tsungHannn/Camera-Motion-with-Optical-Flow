import numpy as np
import matplotlib.pyplot as plt
import cv2


#canny function will contain all the changes
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #original image
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    canny = cv2.Canny(blur, 50,150)
    return canny

# 把Canny畫好的線段畫回去
def overlay_edges(image, lines):
    overlay = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 畫紅色粗線
    return overlay

#Bitwise on the canny edge image
def region_of_int(image):
    height = image.shape[0]
    width = image.shape[1]
    #Coordinates of the triangular region

    bottom_left  = [0, height]
    top_left     = [width*0.35, height*0.5]
    # bottom_right = [cols*0.95, rows]
    bottom_right = [width, height]
    top_right    = [width*0.65, height*0.5] 
    polygons = np.array([
        [bottom_left, top_left, top_right, bottom_right]
    ], dtype=np.int32)

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    roi = image.copy()
    # roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    roi = cv2.polylines(roi,vertices, True, (0,0,255), 10)
    # cv2.imshow("roi", roi)

    # polygons = np.array([
    #     [(0,height * 0.5), (width, height * 0.5), (width, height), (0, height)]
    # ], dtype=np.int32)


    

    #create a black image with the same dimensions as original image  
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image, roi
    
#optimization 

#the coordinates for the averaged out single line
def make_coord(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])
  


#average the slope of multiple lines and make one line
def average_slope(image, lines):
    left_fit= []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2= line.reshape(4)
        
        #fit line to points, return slope and y-int
        parameters = np.polyfit((x1, x2), (y1, y2), 1) 
        slope = parameters[0]
        intercept = parameters[1]
        #lines on the right have positive slope, and lines on the left have negative slope
        if slope < 0:
            left_fit.append((slope, intercept))
        else: 
            right_fit.append((slope, intercept))
    
    #average of all the columns (column0: slope, cloumns: y_int)
    if len(left_fit) == 0 or len(right_fit) == 0:
        # print("No lines detected")
        return np.array([])
    
    left_fit_avg = np.average(left_fit, axis = 0)
    right_fit_avg = np.average(right_fit, axis = 0)
    
    #Create line based on avg value
    left_line = make_coord(image, left_fit_avg)
    right_line = make_coord(image, right_fit_avg)
    return np.array([left_line, right_line])

#Hough transform  
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    
    #to make sure array is not empty
    if lines is not None: 
       for line in lines:
            x1, y1, x2, y2= line.reshape(4)
            
            #Black lines on image
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image 

def find_vanishing_point_by_lane(left_line, right_line):
    if left_line is None or right_line is None or len(left_line) == 0 or len(right_line) == 0:
        return None  # 如果缺少任何一條線，無法計算

    x1_l, y1_l, x2_l, y2_l = left_line
    x1_r, y1_r, x2_r, y2_r = right_line

    # 計算斜率 m 和 截距 b
    m1 = (y2_l - y1_l) / (x2_l - x1_l)
    b1 = y1_l - m1 * x1_l

    m2 = (y2_r - y1_r) / (x2_r - x1_r)
    b2 = y1_r - m2 * x1_r

    # 計算焦點 x, y
    if m1 == m2:  # 避免平行線無法求交點
        return None

    x_vanish = (b2 - b1) / (m1 - m2)
    y_vanish = m1 * x_vanish + b1

    return int(x_vanish), int(y_vanish)


# 整個偵測車道線流程(融合進MV on Vehicle用)
# return (averaged_line, vanishing_point)
# average_line = [left_line, right_line] (可能會只有一條線)  left_line: (x1, y1, x2, y2)
# vanishing_point: (x, y)
def lane_detection(image):
    c1 = canny(image)
    cropped_image, _ = region_of_int(c1)
    _, roi = region_of_int(image)
    # cv2.imshow("roi", roi)
    

    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=50,maxLineGap=10)
    filtered_lines = [] # 濾除水平的線段
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # 計算線段角度
            if abs(angle) > 20:  # 過濾水平線（角度小於20度的線）
                filtered_lines.append([[x1, y1, x2, y2]])
    
    filtered_lines = np.array(filtered_lines)

    gray_with_line = overlay_edges(image, filtered_lines)
    
    averaged_lines = average_slope(image, filtered_lines)

    line_image = display_lines(image, averaged_lines)
    combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)

    vanishing_point = (-1, -1)
    if len(averaged_lines) == 2:  # 確保有兩條車道線
        vanishing_point = find_vanishing_point_by_lane(averaged_lines[0], averaged_lines[1])
        if vanishing_point != (-1, -1):
            cv2.circle(combo_image, vanishing_point, 5, (0, 255, 0), -1)  # 用綠色標示焦點

    # cv2.putText(combo_image, str(frame_id), (590, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    # cv2.imshow("result", combo_image)

    return (combo_image, gray_with_line, averaged_lines, vanishing_point)


if __name__ == "__main__":

    # capture = cv2.VideoCapture(r"/media/mvclab/HDD/mvs_mp4/1220/horizontal_rotation/2024-12-20-06-28-47_mvs_compressed.mp4")
    # capture = cv2.VideoCapture(r"/media/mvclab/HDD/mvs_mp4/0701/gray/test_2024-07-01-02-33-02_mvs_compressed.mp4")
    capture = cv2.VideoCapture("mvs_mp4\\0701\\gray\\test_2024-07-01-02-33-02_mvs_compressed.mp4")
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    frameRate = int(capture.get(cv2.CAP_PROP_FPS))
    outputResult = cv2.VideoWriter("output.mp4", codec, frameRate, (int(capture.get(3)),int(capture.get(4))))
    frame_id = 0

    while(capture.isOpened()):
        ret, frame = capture.read()
        if ret == True:
            c1 = canny(frame)
            
            
            cropped_image = region_of_int(frame)
            
            # cv2.imshow("result", cropped_image)
            # cv2.waitKey(0)
            lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=50,maxLineGap=10)
            filtered_lines = [] # 濾除水平的線段
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # 計算線段角度
                    if abs(angle) > 20:  # 過濾水平線（角度小於20度的線）
                        filtered_lines.append([[x1, y1, x2, y2]])
            
            filtered_lines = np.array(filtered_lines)

            gray_with_edge = overlay_edges(frame, filtered_lines)
            cv2.imshow("line_detection", gray_with_edge)
            # cv2.waitKey(0)


            averaged_lines = average_slope(frame, filtered_lines)
            # try:
            #     averaged_lines = average_slope(frame, filtered_lines)
            # except:
            #     print("No line detected.")
            #     averaged_lines = np.array([])

            line_image = display_lines(frame, averaged_lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

            if len(averaged_lines) == 2:  # 確保有兩條車道線
                vanishing_point = find_vanishing_point(averaged_lines[0], averaged_lines[1])
                if vanishing_point:
                    cv2.circle(combo_image, vanishing_point, 5, (0, 255, 0), -1)  # 用綠色標示焦點

            cv2.putText(combo_image, str(frame_id), (590, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("result", combo_image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            frame_id += 1
            outputResult.write(combo_image)
            if frame_id > 600:
                break
        else:
            break
    capture.release()
    outputResult.release()
    cv2.destroyAllWindows()