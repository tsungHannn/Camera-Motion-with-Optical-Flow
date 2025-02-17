import numpy as np
import cv2 as cv
from sklearn.linear_model import RANSACRegressor
from skimage.draw import line

def create_dense_optical_flow(x_displacement, y_displacement):
        # Normalize displacement maps: [0, 255] -> [-127, 127]
        x_flow = x_displacement.astype(np.float32) - 128
        y_flow = y_displacement.astype(np.float32) - 128
        
        # Combine x and y flows into a dense optical flow representation
        optical_flow = np.zeros((x_displacement.shape[0], x_displacement.shape[1], 2), dtype=np.float32)
        optical_flow[..., 0] = x_flow  # Horizontal displacement (U)
        optical_flow[..., 1] = y_flow  # Vertical displacement (V)
        
        return optical_flow

def visualize_optical_flow(optical_flow):
    # Convert optical flow to HSV image for visualization
    h, w = optical_flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Compute magnitude and angle of the flow
    magnitude, angle = cv.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue: direction
    hsv[..., 1] = 255  # Saturation: max
    hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)  # Value: magnitude
    
    # Convert HSV to RGB for display
    rgb_flow = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return rgb_flow


def generate_flow_legend():
    # Legend dimensions
    legend_size = 300  # Diameter of the legend circle
    radius = legend_size // 2
    center = (radius, radius)
    
    # Create an empty image for the legend
    legend = np.zeros((legend_size, legend_size, 3), dtype=np.uint8)
    
    # Generate HSV values for each pixel
    for y in range(legend_size):
        for x in range(legend_size):
            dx = x - center[0]
            dy = center[1] - y  # Invert y-axis for proper orientation
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance <= radius:
                angle = np.arctan2(dy, dx)  # Angle in radians
                magnitude = distance / radius  # Normalize magnitude
                
                # Convert angle and magnitude to HSV
                hue = ((angle + np.pi) / (2 * np.pi)) * 180  # Convert radians to degrees
                saturation = 255
                value = int(magnitude * 255)  # Normalize to 0-255
                
                # Assign HSV values
                legend[y, x] = (hue, saturation, value)
    
    # Convert HSV to BGR for visualization
    legend_bgr = cv.cvtColor(legend, cv.COLOR_HSV2BGR)
    return legend_bgr


def filter_optical_flow_ransac(optical_flow, sampling_step=10):
    h, w = optical_flow.shape[:2]
    
    # 取得部分光流點 (減少計算量)
    y, x = np.mgrid[0:h:sampling_step, 0:w:sampling_step]
    fx, fy = optical_flow[y, x, 0], optical_flow[y, x, 1]

    # 計算終點 (光流位移後的位置)
    end_x = x + fx
    end_y = y + fy

    # 準備 RANSAC 訓練數據
    X = x.flatten().reshape(-1, 1)  # 影像座標 (輸入)
    Y = end_x.flatten()  # 位移後的 x 座標 (目標)
    
    # 使用 RANSAC 擬合主要方向
    model = RANSACRegressor()
    model.fit(X, Y)
    
    # 預測的光流方向
    predicted_x = model.predict(X)

    # 過濾符合 RANSAC 模型的點 (只保留內部點)
    inlier_mask = model.inlier_mask_
    filtered_x = x.flatten()[inlier_mask]
    filtered_y = y.flatten()[inlier_mask]
    filtered_fx = fx.flatten()[inlier_mask]
    filtered_fy = fy.flatten()[inlier_mask]

    # 建立過濾後的光流場
    filtered_flow = np.zeros_like(optical_flow)
    for i in range(len(filtered_x)):
        filtered_flow[filtered_y[i], filtered_x[i], 0] = filtered_fx[i]
        filtered_flow[filtered_y[i], filtered_x[i], 1] = filtered_fy[i]

    return filtered_flow

def find_vanishing_point_by_optical_flow(optical_flow):
    h, w = optical_flow.shape[:2]
    
    # 取得光流的端點 (起點固定於像素中心)
    y, x = np.mgrid[0:h, 0:w]
    fx, fy = optical_flow[..., 0], optical_flow[..., 1]

    # 終點座標
    end_x = x + fx
    end_y = y + fy

    # 建立可視化用的空白圖
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # 繪製流場線段
    for i in range(0, h, 10):  # 每隔 10 像素取一點
        for j in range(0, w, 10):
            pt1 = (j, i)
            pt2 = (int(end_x[i, j]), int(end_y[i, j]))
            cv.arrowedLine(vis, pt1, pt2, (255, 255, 255), 1)

    # 轉換為灰階，進行霍夫變換
    gray = cv.cvtColor(vis, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    
    # 計算線的交點 (使用最小二乘法計算最可能的交點)
    points = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            for line2 in lines:
                x3, y3, x4, y4 = line2[0]
                if x1 == x3 and y1 == y3:  # 避免同一條線重複計算
                    continue
                
                # 計算兩條線的交點
                A1, B1, C1 = y2 - y1, x1 - x2, x2 * y1 - x1 * y2
                A2, B2, C2 = y4 - y3, x3 - x4, x4 * y3 - x3 * y4
                
                det = A1 * B2 - A2 * B1
                if abs(det) < 1e-5:  # 平行線跳過
                    continue
                
                px = (B1 * C2 - B2 * C1) / det
                py = (C1 * A2 - C2 * A1) / det
                
                if 0 <= px < w and 0 <= py < h:  # 確保點在畫面內
                    points.append((int(px), int(py)))

    # 如果找到了交點，計算平均交點作為消失點
    if points:
        vanishing_point = np.mean(points, axis=0).astype(int)
    else:
        vanishing_point = (-1, -1)

    return vanishing_point, vis, points



def draw_infinite_lines(flow):
    height, width, _ = flow.shape
    line_count = np.zeros((height, width), dtype=np.int32)
    line_visual = np.zeros((height, width, 3), dtype=np.uint8)
    
    y, x = np.mgrid[0:height, 0:width]
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    
    for i in range(height):
        for j in range(width):
            dx, dy = flow_y[i, j], flow_x[i, j]
            if dx == 0 and dy == 0:
                continue
            
            # Compute line equation y = ax + b
            if abs(dx) > abs(dy):
                a = dy / dx
                for x_p in range(width):
                    y_p = int(i + a * (x_p - j))
                    if 0 <= y_p < height:
                        line_count[y_p, x_p] += 1
                        line_visual[y_p, x_p] = (0, 255, 0)  # Green line
            else:
                a = dx / dy
                for y_p in range(height):
                    x_p = int(j + a * (y_p - i))
                    if 0 <= x_p < width:
                        line_count[y_p, x_p] += 1
                        line_visual[y_p, x_p] = (0, 255, 0)  # Green line
    
    return line_count, line_visual


