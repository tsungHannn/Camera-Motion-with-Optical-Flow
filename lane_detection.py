import numpy as np
import matplotlib.pyplot as plt
import cv2


#canny function will contain all the changes
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #original image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50,150)
    return canny

#Bitwise on the canny edge image
def region_of_int(image):
    height = image.shape[0]
    width = image.shape[1]
    #Coordinates of the triangular region
    polygons = np.array([
        [(0,height * 0.5), (width, height * 0.5), (width, height), (0, height)]
    ], dtype=np.int32)
    #create a black image with the same dimensions as original image  
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
    
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
        print("No lines detected")
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


capture = cv2.VideoCapture(r"/media/mvclab/HDD/mvs_mp4/1220/horizontal_rotation/2024-12-20-06-28-47_mvs_compressed.mp4")
# capture = cv2.VideoCapture(r"/media/mvclab/HDD/mvs_mp4/0701/gray/test_2024-07-01-02-33-02_mvs_compressed.mp4")
frame_id = 0
while(capture.isOpened()):
    ret, frame = capture.read()
    if ret == True:
        c1 = canny(frame)
        # cropped_image = region_of_int(c1)
        # cv2.imshow("result", cropped_image)
        # cv2.waitKey(0)
        lines = cv2.HoughLinesP(c1, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)

        averaged_lines = average_slope(frame, lines)
        if len(averaged_lines) == 0:
            continue
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.putText(combo_image, str(frame_id), (590, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("result", combo_image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        frame_id += 1
    else:
        break
capture.release()
cv2.destroyAllWindows()