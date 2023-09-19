
import pygame
import numpy as np
import heapq

from itertools import permutations
import cv2
import time
import cv2
import numpy as np
import math
import threading


approxPolyDP_epslion=0.043#多边形近似参数，越小越精准
normal_std=0.8#判断四条边的归一标准差
min_center_distance=10#中心距离
wh_rate=0.67#长宽比系数，越小越接近正方形
min_area=20
max_area=4000

min_ridus=8#最小半径
max_ridus=22#最大半径
minDist =40#第四个参数是两个圆之间的最小距离。
param1  =50#第五个参数是Canny边缘检测的高阈值，它用于检测图像中的边缘。
param2  =16#第六个参数是累加器阈值。值越小，检测到的圆形越少。

# 已知点和目标点
src = np.array([(1, 1), (10, 1), (10, 10), (1, 10)], dtype=np.float32)
dst = np.array([(19, 1), (19, 19), (1, 19), (1,1)], dtype=np.float32)

# 计算仿射变换矩阵
M = cv2.getPerspectiveTransform(src, dst)

# 输出变换矩阵
def apply_affine_transform(src, M):
    """
    使用仿射变换矩阵将点 src 变换为目标点
    
    参数：
    - src：原始点坐标，形状为 (N, 2)
    - M：仿射变换矩阵，形状为 (3, 3)
    
    返回值：
    - dst：目标点坐标，形状为 (N, 2)
    """
    # 增加一维，将 src 变为 (N, 3) 的矩阵
    src_3d = np.hstack([src, np.ones((src.shape[0], 1))])
    
    # 使用仿射变换矩阵变换 src
    dst_3d = np.matmul(M, src_3d.T).T
    
    # 将 dst_3d 的最后一列去掉，得到目标点坐标
    dst = dst_3d[:, :2]
    
    return dst
def apply_inverse_affine_transform(dst, M):
    """
    使用仿射变换矩阵的逆变换将目标点 dst 变换回原始点
    
    参数：
    - dst：目标点坐标，形状为 (N, 2)
    - M：原始点到目标点的仿射变换矩阵，形状为 (3, 3)
    
    返回值：
    - src：原始点坐标，形状为 (N, 2)
    """
    # 计算仿射变换矩阵的逆矩阵
    M_inv = np.linalg.inv(M)
    
    # 增加一维，将 dst 变为 (N, 3) 的矩阵
    dst_3d = np.hstack([dst, np.ones((dst.shape[0], 1))])
    
    # 使用仿射变换矩阵的逆变换将 dst 变换为 src
    src_3d = np.matmul(M_inv, dst_3d.T).T
    
    # 将 src_3d 的最后一列去掉，得到原始点坐标
    src = src_3d[:, :2]
    return src
def p2m(x,y):#10*10到21*21
    new_x,new_y=apply_affine_transform(np.array([(x, y)]), M)[0]
    new_x,new_y=round(new_x),round(new_y)
    return new_x,new_y
def m2p(x,y):#10*10到21*21
    new_x,new_y=apply_inverse_affine_transform(np.array([(x, y)]), M)[0]
    new_x,new_y=round(new_x,1),round(new_y,1)
    return new_x,new_y

def A_star2(map1, start, end):
    """
    使用A*算法寻找起点到终点的最短路径
    :param map: 二维列表，表示地图。0表示可以通过的点，1表示障碍物。
    :param start: 元组，表示起点坐标。
    :param end: 元组，表示终点坐标。
    :return:path :表示从起点到终点的最短路径，其中每个元素是一个坐标元组。
            path_length:路径的长度
    """
    # 定义启发式函数（曼哈顿距离）
    def heuristic(node1, node2):
        return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

    # 初始化open_list、closed_list、g_score、came_from
    open_list = [(0, start)]
    closed_list = set()
    g_score = {start: 0}
    came_from = {}
    path_length=0
    # 开始搜索
    while open_list:
        # 取出f值最小的节点
        current = heapq.heappop(open_list)[1]
        if current == end:
            # 找到终点，返回路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path_length=int((len(path)-1)/2)
            return path[::-1],path_length
            
        # 将当前节点加入closed_list
        closed_list.add(current)

        # 遍历相邻节点
        for neighbor in [(current[0] - 1, current[1]),
                         (current[0] + 1, current[1]),
                         (current[0], current[1] - 1),
                         (current[0], current[1] + 1)]:
            if 0 <= neighbor[0] < len(map1) and 0 <= neighbor[1] < len(map1[0]) and map1[neighbor[0]][neighbor[1]] == 0:
                # 相邻节点是可通过的节点
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # 如果相邻节点不在g_score中，或者新的g值更优，则更新g_score和came_from
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_list, (f_score, neighbor))
                    came_from[neighbor] = current
    
    
    # 没有找到可行路径，返回空列表
    return [],path_length
def precomputation(map1,start,end,mid_points):
    x=mid_points[:]
    x.append(start)
    x.append(end)
    permutations_list = list(permutations(x,2))
    length_dict={}
    for pt in x:
        length_dict[pt]={}
    for pt1,pt2 in permutations_list:
        length_dict[pt1][pt2]=A_star2(map1,pt1,pt2)[1]
    return length_dict
def get_min_path(map1,start,end,mids):
    """
    返回21*21地图坐标路径点和路径长度
    """
    #穷举法8！=40320
    #计算1000个路径需要3s，全部计算需要2分钟计算太慢,但是使用路径查询后大大减少了计算量40320组数据在0.2s完成计算获得最优路径
    length_dict=precomputation(map1,start,end,mids)
    
    permutations_list = list(permutations(mids))
    #print(permutations_list)
    min_path_length=float("inf")
    min_path=None
    all_points=[]
    for mid_points in permutations_list:
        mid_points=list(mid_points)
        #print("mid_points",mid_points)
        mid_points.append(end)
        mid_points.insert(0,start)
        #print("mid_points2",mid_points)
        all_length=0
        for i in range(len(mid_points)-1):
            if length_dict:#如果没有预计算则采用现场计算，很费时
                length=length_dict[mid_points[i]][mid_points[i+1]]
                #print(length_dict,mid_points[i],[mid_points[i+1]])
            else:
                length=A_star2(map1,mid_points[i],mid_points[i+1])[1]
            all_length+=length
        if all_length<min_path_length:
            min_path_length=all_length
            min_path=mid_points
    
    for i in range(len(min_path)-1):
        for j in A_star2(map1,min_path[i],min_path[i+1])[0][:-1]:
            all_points.append(j)
    all_points.append(min_path[-1])
        
    return all_points,min_path_length 

def path_plan(map1,start,end,mids):
    
    #start=(1,1)
    #end=(10,10)
    #mids=[(5,5),(6,6)]
    path=get_min_path(map1,p2m(*start),p2m(*end),[p2m(*i) for i in mids])[0]
    all_points=[]
    for i in path:
        all_points.append(m2p(*i))
    return all_points
def generate_border_pygame(color=(0,0,0),line_scale=5):
    
    list1=[
(0 ,0 ,10, 0),
(10, 0, 10 ,9),
(0 ,1 ,1,1),
(2 ,0 ,2 ,3),
(1 ,2 ,2 ,2),
(2 ,3 ,3 ,3),
(3 ,3 ,3 ,1),
(3 ,1 ,4 ,1),
(5 ,0 ,5 ,1),
(4 ,2 ,5 ,2),
(6 ,1 ,6 ,3),
(6 ,2 ,7 ,2),
(4 ,3 ,6 ,3),
(4 ,3 ,4 ,4),
(3 ,4 ,6 ,4),
(0,3,1,3),
(1,3,1,4),
(1,4,2,4),
(1,5,3,5),
(1,5,1,6),
(1,6,2,6),
(2,6,2,7),
(3,5,3,7),
(4 ,5, 5 ,5),
(0 ,7, 1 ,7),
(1 ,7, 1 ,9),
(1 ,8, 2 ,8),
(2 ,8, 2 ,9),
(2 ,9, 3 ,9),
]
    for i in list1:
        start_x, start_y, end_x, end_y = i

        start_x2, start_y2, end_x2, end_y2=10-start_x, 10-start_y, 10-end_x, 10-end_y
        # 将坐标转换为整数
        start_x, start_y, end_x, end_y = int(start_x)*50+150, 650-int(start_y)*50, int(end_x)*50+150, 650-int(end_y)*50


        start_x2, start_y2, end_x2, end_y2=int(start_x2)*50+150, 650-int(start_y2)*50, int(end_x2)*50+150, 650-int(end_y2)*50

        # 绘制直线
        pygame.draw.line(screen, color,(start_x, start_y),  (end_x, end_y), line_scale)
        pygame.draw.line(screen, color,(start_x2, start_y2),  (end_x2, end_y2), line_scale)

def pixel2pose(pixel_x,pixel_y):
    """将像素坐标中的点转换到10*10坐标中"""
    pose_x=round((pixel_x-125)/50)
    pose_y=11-round((pixel_y-125)/50)
    return pose_x,pose_y

def pose2pixel(pose_x,pose_y):
    """
    函数接受两个参数，分别为机器人的位置坐标pose_x和pose_y，返回值为该位置对应的图像像素坐标pixel_x和pixel_y。
    """
    pixel_x=125+50*pose_x
    pixel_y=675-50*pose_y
    return pixel_x,pixel_y


def draw_paths(paths,color=(139,0,139),line_scale=3):
    for i in range(len(paths)-1):
        start_x, start_y=paths[i]
        end_x, end_y =paths[i+1]

        #start_x2, start_y2, end_x2, end_y2=10-start_x, 10-start_y, 10-end_x, 10-end_y
        # 将坐标转换为整数
        start_x, start_y, =pose2pixel(start_x, start_y)
        end_x, end_y = pose2pixel(end_x, end_y)


        #start_x2, start_y2, end_x2, end_y2=int(start_x2)*50+150, 650-int(start_y2)*50, int(end_x2)*50+150, 650-int(end_y2)*50
        # 绘制直线
        pygame.draw.line(screen, color,(start_x, start_y),  (end_x, end_y), line_scale)
        #pygame.draw.line(screen, color,(start_x2, start_y2),  (end_x2, end_y2), line_scale)
    color1 = (255, 0, 0)#起始点
    color2 = (255,0,255)#中间点
    color3 = (0, 0, 255)#末尾点
    text1 = font.render("path_length=%d"%((len(paths)-1)/2), True, (0, 0, 0))
    text_rect1 = text1.get_rect()
    text_rect1.centerx = 400
    text_rect1.centery =50
    #print(text_rect1.centerx,text_rect1.centery )
    screen.blit(text1, text_rect1)
    text1 = font.render("start_point", True, color1)
    text_rect1 = text1.get_rect()
    text_rect1.centerx =200
    text_rect1.centery =700
    #print(text_rect1.centerx,text_rect1.centery )
    screen.blit(text1, text_rect1)
    text1 = font.render("end_point", True, color3)
    text_rect1 = text1.get_rect()
    text_rect1.centerx = 200
    text_rect1.centery =730
    #print(text_rect1.centerx,text_rect1.centery )
    screen.blit(text1, text_rect1)
    text1 = font.render("intermediate_points", True, color2)
    text_rect1 = text1.get_rect()
    text_rect1.centerx = 200
    text_rect1.centery =760
    #print(text_rect1.centerx,text_rect1.centery )
    screen.blit(text1, text_rect1)

def draw_points(points):
    
    radius1 = 10#起始，末尾
    radius2 = 8#中间
    color1 = (255, 0, 0)#起始点
    color2 = (255,0,255)#中间点
    color3 = (0, 0, 255)#末尾点


    if len(points)==0:
        pass
    
    elif len(points)==1:
        pygame.draw.circle(screen, color1, pose2pixel(*points[0]), radius1)

    elif len(points)==2:
        pygame.draw.circle(screen, color1, pose2pixel(*points[0]), radius1)
        pygame.draw.circle(screen, color3, pose2pixel(*points[-1]), radius1)

    else:
        for index,point in enumerate(points[1:-1]):
            pygame.draw.circle(screen, color2, pose2pixel(*point), radius2)

        pygame.draw.circle(screen, color1, pose2pixel(*points[0]), radius1)
        pygame.draw.circle(screen, color3, pose2pixel(*points[-1]), radius1)
    for index,point in enumerate(points):
        text1 = font.render("%d"%(index), True, (0, 0, 255))
        text_rect1 = text1.get_rect()
        text_rect1.centerx = pose2pixel(*points[index])[0]+20
        text_rect1.centery = pose2pixel(*points[index])[1]
        #print(text_rect1.centerx,text_rect1.centery )
        screen.blit(text1, text_rect1)


def pose2pixel(pose_x,pose_y):
    """
    函数接受两个参数，分别为机器人的位置坐标pose_x和pose_y，返回值为该位置对应的图像像素坐标pixel_x和pixel_y。
    """
    pixel_x=125+50*pose_x
    pixel_y=675-50*pose_y
    return pixel_x,pixel_y

def standardize(data):
    """
    对数据进行离差标准化
    """
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

def std_dev(data):
    """
    计算标准差
    """
    mean_val = np.mean(data)
    deviation = data - mean_val
    return np.sqrt(np.sum(deviation ** 2) / len(data))

def distance(point1, point2):
    """计算距离"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def filter_points(points, threshold):
    """
    - points：包含坐标信息的列表
    - threshold：距离阈值，当两个坐标之间的距离小于此值时，剔除其中的一个坐标
    返回值：selected_points，包含经过筛选后的坐标信息的列表
    功能：对传入的坐标列表进行筛选，去掉距离过近的坐标，返回经过筛选后的坐标信息列表
    实现思路：对于每一个坐标，遍历其后面的每一个坐标，如果有一个坐标与当前坐标距离小于阈值，则剔除当前坐标，
    继续遍历下一个坐标，如果所有后面的坐标与当前坐标的距离都大于等于阈值，则将当前坐标添加到
    筛选后的坐标列表中，遍历完所有坐标后返回筛选后的坐标列表。
    """
    selected_points = []  # 定义一个空列表，存储筛选出来的坐标
    for i in range(len(points)):
        is_selected = True
        for j in range(i+1, len(points)):
            if distance(points[i], points[j]) < threshold:
                is_selected = False
                break
        if is_selected:
            selected_points.append(points[i])
    return selected_points

def find_locating_box(frame,display=False):
    """
    该函数主要功能是在给定图像中寻找包含目标物体的定位框（location box）。函数接收两个参数，第一个是待处理的图像帧，第二个是一个可选的布尔类型参数，用于指定是否显示处理过程的可视化结果。
    该函数具体的处理流程如下：
    将图像帧转换为灰度图像，再使用Canny边缘检测算法获得边缘图像。
    在边缘图像中寻找轮廓。
    根据预设的筛选条件对轮廓进行筛选，得到一组可能的四边形。
    根据预设的条件，从可能的四边形中筛选出正方形。
    根据预设的条件，从正方形中筛选出符合包含关系的正方形，即为定位框。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #筛选条件0 面积
    contours2=[]
    for i in range(len(contours)):
        M1 = cv2.moments(contours[i])
        if M1['m00']<min_area or M1['m00']>max_area:
            continue
        contours2.append(contours[i])

    
    #筛选条件1 四边形
    sibianxing=[]
    for c in contours2:
        # 对轮廓进行多边形逼近
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, approxPolyDP_epslion * peri, True)
        hull=approx
        if len(hull)==4:
            sibianxing.append(hull)

    
    #筛选条件2 正方形
    #判断正方形方案1，判断长宽比
    zhengfangxing=[]
    for hull in sibianxing:

        rect = cv2.minAreaRect(hull)
        (x, y), (w, h), angle = rect
        abs(w - h)
        if abs(w - h)/(w+h)<wh_rate:#边长归一标准差
            zhengfangxing.append(hull)

    #筛选条件3 正方形的包含关系
    dingweikuang=[]
    for i in range(len(zhengfangxing)-1):
        M1 = cv2.moments(zhengfangxing[i])
        if M1['m00']==0:
             continue
        cx1 = int(M1['m10'] / M1['m00'])
        cy1= int(M1['m01'] / M1['m00'])
        for j in range(i+1,len(zhengfangxing)):
            M2 = cv2.moments(zhengfangxing[j])
            if M2['m00']==0:#两者面积为0则跳过
                continue
            cx2 = int(M2['m10'] / M2['m00'])
            cy2 = int(M2['m01'] / M2['m00'])
            
            #1二者中心互相包含
            res1=cv2.pointPolygonTest(zhengfangxing[i],(cx2,cy2), False)
            res2=cv2.pointPolygonTest(zhengfangxing[j],(cx1,cy1), False)
            if res1>0 and res2>0:#如果两个轮廓中心具有包含关系         
                #2计算中心距离
                distance_=distance((cx1,cy1),(cx2,cy2))
                if distance_<min_center_distance:#中心距里小于5个像素      
                    dingweikuang+=[zhengfangxing[i],zhengfangxing[j]]
    if display:
        cv2.imshow('step1_edges', edges)
        
        contours_frame=frame.copy()
        cv2.drawContours(contours_frame,contours, -1, (0, 0, 255), 1)
        cv2.putText(contours_frame, "counts:{}".format(len(contours)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('step2_all_contours', contours_frame)
       
        contours2_frame=frame.copy()
        cv2.drawContours(contours2_frame,contours2, -1, (0, 0, 255), 1)
        "FPS: {:.2f}".format(fps)
        cv2.putText(contours2_frame, "min_area:{},max_area:{},counts:{}".format(min_area,max_area,len(contours2)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('step3_Area_Selected_contours', contours2_frame)  
        
        sibianxing_frame=frame.copy()
        cv2.drawContours(sibianxing_frame,sibianxing, -1, (0, 0, 255), 1)
        cv2.putText(sibianxing_frame, "approxPolyDP_epslion:{:.2f},counts:{}".format(approxPolyDP_epslion,len(sibianxing)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('step4_quadrilateral_selection', sibianxing_frame)  
        
        zhengfangxing_frame=frame.copy()
        cv2.drawContours(zhengfangxing_frame,zhengfangxing, -1, (0, 0, 255), 1)
        cv2.putText(zhengfangxing_frame, "wh_rate:{:.2f},counts:{}".format(wh_rate,len(zhengfangxing)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('step5_Square_selection', zhengfangxing_frame)  
        
        dingweikuang_frame=frame.copy()
        cv2.drawContours(dingweikuang_frame,dingweikuang, -1, (0, 0, 255), 1)
        cv2.putText(dingweikuang_frame, "min_center_distance:{:.2f},counts:{}".format(min_center_distance,len(dingweikuang)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('step6_Positioning_box_selection', dingweikuang_frame) 
        
    # 返回帧
    #print(len(contours),len(sibianxing),len(zhengfangxing),len(dingweikuang))
    return dingweikuang

def get_corners_direction(corners):
    """判断四个角点的方向"""
    x_avg = sum(p[0] for p in corners) / 4  # 计算四个角点的x坐标平均值
    y_avg = sum(p[1] for p in corners) / 4  # 计算四个角点的y坐标平均值
    tl, tr, bl, br = None, None, None, None
    for p in corners:
        if p[0] < x_avg and p[1] < y_avg:
            tl = p
        elif p[0] > x_avg and p[1] < y_avg:
            tr = p
        elif p[0] < x_avg and p[1] > y_avg:
            bl = p
        elif p[0] > x_avg and p[1] > y_avg:
            br = p
    return tl, tr, bl, br

def get_locating_point(locating_boxs):
    """
    这个函数名为 get_locating_point，它的作用是获取定位框的中心点。传入的参数 locating_boxs 是一个包含若干个定位框的列表，每个定位框是一个由点坐标构成的列表。
    该函数首先定义了一个空列表 dingweidian 和一个空列表 locating_points。然后遍历传入的所有定位框，通过 OpenCV 的 cv2.moments 函数计算每个定位框的中心点坐标，并将其添加到 dingweidian 列表中。
    接着调用了 filter_points 函数对 dingweidian 中的点进行筛选，把距离较近的点合并成一个点，最终将得到的筛选后的点添加到 locating_points 列表中。
    最后返回 locating_points 列表，其中包含了所有定位框的中心点坐标。
    """
    # 定义两个空列表
    dingweidian=[]
    locating_points=[]
    # 循环定位框列表中的每一个框
    for i in range(len(locating_boxs)):
        # 计算当前框的矩（moments）
        M1 = cv2.moments(locating_boxs[i])
        # 计算当前框的重心坐标
        cx1 = int(M1['m10'] / M1['m00'])
        cy1= int(M1['m01'] / M1['m00'])
        # 将当前框的重心坐标添加到列表中
        dingweidian.append((cx1,cy1))
    # 将中心距里较近的点合并成一个点
    locating_points=filter_points(dingweidian,10)
    # 返回合并后的点列表
    return locating_points

def image_correction(img,locating_points):
    """tl, tr, bl, br"""
    """该函数名为image_correction，用于对输入的图像进行校正，使其四个角点的位置分别接近预设的目标位置。该函数需要输入两个参数：img为待校正的图像，locating_points为检测出的定位框中心点坐标列表。
    函数会将定位框中心点坐标列表locating_points与预设目标位置坐标列表goal_box_centers进行透视变换，以得到透视矫正后的图像。如果矫正失败，函数会返回两个None值。
    函数返回两个值：warped_img为矫正后的图像，M为变换矩阵。
    """
    real_box_centers=locating_points
    goal_box_centers=[(75,75),(725,75),(75,725),(725,725)]
    if None in real_box_centers:#矫正失败
        return None,None
    dst_pts = np.array(goal_box_centers, dtype=np.float32)
    src_pts = np.array(real_box_centers, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(img, M, (800,800))
    return warped_img,M

def generate_border(img,color=(0,0,0),line_scale=3):
    """generate_border 要求传入的输入图片为800*800"""
    x_start = 400 - (10 // 2) * 50
    y_start = 400 - (10 // 2) * 50
    x_end = 400 + (10 // 2) * 50
    y_end = 400 + (10 // 2) * 50

    # 绘制水平方向的网格线
    for i in range(11):
        y = y_start + i * 50
        cv2.line(img, (x_start, y), (x_end, y), color, line_scale)

    # 绘制垂直方向的网格线
    for j in range(11):
        x = x_start + j * 50
        cv2.line(img, (x, y_start), (x, y_end), color, line_scale)
    
    cv2.rectangle(img, (0, 580), (170, 800), (0, 0, 0), -1)#z左下遮挡
    cv2.rectangle(img, (630, 0), (800, 220), (0, 0, 0), -1)#右上遮挡
    cv2.rectangle(img, (0, 0), (170, 170), (0, 0, 0), -1)#左上遮挡
    cv2.rectangle(img, (630, 630), (800,800), (0, 0, 0), -1)    
    return img

def pixel2pose(pixel_x,pixel_y):
    """将像素坐标中的点转换到10*10坐标中"""
    pose_x=round((pixel_x-125)/50)
    pose_y=11-round((pixel_y-125)/50)
    return pose_x,pose_y

def treasure_Identification2(img,display=False):
    """掩膜轮廓法返回宝藏的像素坐标"""
    """函数treasure_Identification2是用掩膜轮廓法从图像中获取宝藏的像素坐标。它的参数是一个图像和一个布尔值，控制是否显示绘制有宝藏像素点的图像。函数首先对图像进行了一些处理，包括将其转换为灰度图像、对其进行二值化操作和消除边框。然后，使用cv2.findContours函数找到图像中所有的轮廓，并对这些轮廓进行循环。对于每个轮廓，函数计算出其中心坐标，并检查其是否在特定区域内。如果是，则将其中心坐标添加到列表centers0中。最后，如果display为True，则将绘制有宝藏像素点的图像显示出来，并返回centers0列表。否则，仅返回centers0列表。"""
    warped_img=img.copy()
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, tresh_warped = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    without_border_image=generate_border(tresh_warped,line_scale=15)
    kernel = np.ones((5,5),np.uint8)
    # 进行开运算操作
    opening = cv2.morphologyEx(without_border_image, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('opening Picture after removing border', opening)
    
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers0=[]
    
    for hull in contours:
        M = cv2.moments(hull)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if 150<cx<650 and 150<cy<650:#特定区域
            centers0.append((cx,cy))
    if display:
        for i in centers0:
            cv2.circle(warped_img,i,10,color=(0,0,255),thickness=3)
        cv2.imshow('treasure_Identification2', warped_img)
    
    return centers0

def treasure_Identification1(img,display=False):
    """霍夫圆检测法返回宝藏的像素坐标"""
    """这个函数使用了霍夫圆检测法来检测宝藏的像素坐标，并返回宝藏中心的像素坐标列表。其中，使用cv2.HoughCircles函数进行霍夫圆检测，设置了一系列参数（minDist、param1、param2、min_ridus和max_ridus）来控制检测效果。如果检测到宝藏，就将中心点坐标加入到centers0列表中，并可以选择在图像上绘制出圆形和参数信息。
    函数的参数为img和display（默认为False），img是输入的图像，display用于控制是否在图像上绘制圆形和参数信息。函数返回centers0，即宝藏中心的像素坐标列表。"""
    #min_ridus=8#最小半径
    #max_ridus=22#最大半径
    #minDist =40#第四个参数是两个圆之间的最小距离。
    #param1  =50#第五个参数是Canny边缘检测的高阈值，它用于检测图像中的边缘。
    #param2  =16#第六个参数是累加器阈值。值越小，检测到的圆形越少。
    warped_img=img.copy()
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=minDist, param1=param1,param2=param2, minRadius=min_ridus, maxRadius=max_ridus)
    centers0=[]
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if 150<x<650 and 150<y<650: 
                if display:
                    cv2.circle(warped_img, (x, y), r, (0, 255, 0), 1)
                centers0.append((x,y))
    if display:
        cv2.putText(warped_img,"minDist,param1,param2,min_ridus,max_ridus={},{},{},{},{}".format(minDist,param1,param2,min_ridus,max_ridus), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('step7_treasure_Identification', warped_img)
    return centers0

def on_trackbar(val):
    
    global approxPolyDP_epslion,normal_std,min_center_distance,wh_rate,min_area,max_area,min_ridus,max_ridus,minDist,param1 ,param2 
    approxPolyDP_epslion = cv2.getTrackbarPos('approxPolyDP_epslion', 'image')*0.001
    wh_rate=cv2.getTrackbarPos('wh_rate', 'image')*0.01
    #normal_std = cv2.getTrackbarPos('normal_std', 'image')*0.01
    min_center_distance = cv2.getTrackbarPos('min_center_distance', 'image')
    min_area=cv2.getTrackbarPos('min_area', 'image')
    max_area=cv2.getTrackbarPos('max_area', 'image')
    
    minDist=cv2.getTrackbarPos('minDist', 'image')
    param1=cv2.getTrackbarPos('param1', 'image')
    param2=cv2.getTrackbarPos('param2', 'image')
    min_ridus=cv2.getTrackbarPos('min_ridus', 'image')
    max_ridus=cv2.getTrackbarPos('max_ridus', 'image')
    
def find_symmetric_point(x1, y1, x2, y2, x3, y3):
    """find_symmetric_point 已知矩形三个点求第四个点
    """
    
    # Calculate the length of each side
    AB = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    BC = math.sqrt((x3-x2)**2 + (y3-y2)**2)
    AC = math.sqrt((x3-x1)**2 + (y3-y1)**2)

    # Find the longest side and its endpoints
    if AB > BC and AB > AC:
        x_long1, y_long1 = x1, y1
        x_long2, y_long2 = x2, y2
        x_other, y_other = x3, y3
    elif BC > AB and BC > AC:
        x_long1, y_long1 = x2, y2
        x_long2, y_long2 = x3, y3
        x_other, y_other = x1, y1
    else:
        x_long1, y_long1 = x1, y1
        x_long2, y_long2 = x3, y3
        x_other, y_other = x2, y2

    # Find the midpoint of the longest side
    x_mid = (x_long1 + x_long2) / 2
    y_mid = (y_long1 + y_long2) / 2


    d_x=x_mid-x_other
    d_y=y_mid-y_other
    
    x_sym=x_long1+x_long2-x_other
    y_sym=y_long1+y_long2-y_other
    # Return the coordinates of the symmetric point
    return x_sym, y_sym

def get_maze_map_pose(img,display=False):
    """输入图像，输出宝藏坐标和标记后的图像"""
    map_maze_loaction=[]
    frame=img.copy()
    dingweikuang =find_locating_box(frame,display=display)
    locating_points=get_locating_point(dingweikuang)
    if len(locating_points)==4:
        locating_points=get_corners_direction(locating_points)
        warped_img,M=image_correction(frame,locating_points)
        if warped_img is not None:
            M_inv = cv2.invert(M)[1]
            maze_piexl_location=treasure_Identification1(warped_img,display=display)#霍夫圆检测法
            for i in range(len(maze_piexl_location)):
                cv2.circle(warped_img,maze_piexl_location[i],10,color=(0,0,255),thickness=3)
                map_maze_loaction.append(pixel2pose(*maze_piexl_location[i]))
            #print("宝藏位置",map_maze_loaction)
            
            origin_maze_pixel=[]
            for pt in maze_piexl_location:
                p_prime=np.array([pt[0], pt[1], 1])
                p = np.dot(M_inv, p_prime)
                # 将齐次坐标转换为笛卡尔坐标
                x, y, w = p
                x /= w
                y /= w
                x=int(x)
                y=int(y)
                origin_maze_pixel.append((x,y))
            for i in range(len(origin_maze_pixel)):
                cv2.circle(frame,origin_maze_pixel[i],5,color=(0,0,255),thickness=2)
            cv2.drawContours(frame,dingweikuang, -1, (0, 255, 0), 1)
    return map_maze_loaction,frame

def online_parameter():
    cv2.namedWindow('image')
    cv2.createTrackbar('min_area', 'image', 1, 100, on_trackbar)
    cv2.setTrackbarPos("min_area",'image',10)
    cv2.createTrackbar('max_area', 'image', 200, 10000, on_trackbar)
    cv2.setTrackbarPos("max_area",'image',4000)
    cv2.createTrackbar('approxPolyDP_epslion', 'image', 1, 100, on_trackbar)
    cv2.setTrackbarPos("approxPolyDP_epslion",'image',43)
    cv2.createTrackbar('wh_rate', 'image', 1, 100, on_trackbar)
    cv2.setTrackbarPos("wh_rate",'image',67)
    #cv2.createTrackbar('normal_std', 'image', 1, 100, on_trackbar)
    #cv2.setTrackbarPos("normal_std",'image',50)
    cv2.createTrackbar('min_center_distance', 'image', 1, 100, on_trackbar)
    cv2.setTrackbarPos("min_center_distance",'image',10)


    cv2.createTrackbar('min_ridus', 'image', 1, 100, on_trackbar)
    cv2.setTrackbarPos("min_ridus",'image',8)
    cv2.createTrackbar('max_ridus', 'image', 30, 100, on_trackbar)
    cv2.setTrackbarPos("max_ridus",'image',22)
    cv2.createTrackbar('minDist', 'image', 1, 200, on_trackbar)
    cv2.setTrackbarPos("minDist",'image',40)
    cv2.createTrackbar('param1', 'image', 1, 100, on_trackbar)
    cv2.setTrackbarPos("param1",'image',50 )
    cv2.createTrackbar('param2', 'image', 1, 100, on_trackbar)
    cv2.setTrackbarPos("param2",'image',16 )

def draw_screen():
    global current_locations
    while True:
        if current_locations and len(current_locations)<10:#避免太多点卡住
            paths=path_plan(map1,(1,1),(10,10),current_locations)
            print(paths)
            screen.fill(bg_color)
            generate_border_pygame()
            #draw_points([(1,1)]+current_locations+[(10,10)])#绘制点
            #draw_points(current_points)#绘制点
            draw_paths(paths)#绘制路线
            pygame.display.flip()
        
        time.sleep(.1)
    
if __name__=="__main__":
    map1=[
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1],
[1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1],
[1,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1],
[1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1],
[1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1],
[1,1,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,1],
[1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1],
[1,0,1,1,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1],
[1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1],
[1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1],
[1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1],
[1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1],
[1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1],
[1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1],
[1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1],
[1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1],
[1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,1,0,1],
[1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

    pygame.init()
    # 设置直线的颜色
    line_color = (255, 0, 0)
    font = pygame.font.Font(None, 36)
    # 设置窗口大小
    size = (800, 800)
    screen = pygame.display.set_mode(size)
    # 设置窗口标题
    pygame.display.set_caption("路径规划演示")
    # 设置背景颜色为白色
    bg_color = (255, 255, 255)
    # 刷新窗口
    screen.fill(bg_color)
    generate_border_pygame()
    pygame.display.flip()
    # 事件循环
    current_locations=[]
    current_points=[]
    #是否展示中间过程调整参数TRUE
    change_parameter=False#False True
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = cap.read()
    if change_parameter:
        online_parameter()

    last_time=time.time()
    last_loction=[]
    frame_count=0
    fps=0
    success_count=0
    success_rate=0
    
    threading.Thread(target=draw_screen).start()
    while True:
        # 获取一帧视频
        ret, frame = cap.read()
        frame_count+=1
        if ret:
            location,img=get_maze_map_pose(frame,display=change_parameter)
            location=set(location)
            if last_loction  and location==last_loction:
                success_count+=1
            last_loction=location
            current_locations=list(location)

            if frame_count==20:
                now_time=time.time()
                during=now_time-last_time
                last_time=now_time
                fps = 20/ during
                frame_count=0
                success_rate=success_count/20
                success_count=0
            print(location,round(fps,2),success_rate)
            cv2.putText(img, "FPS: {:.2f},Success_rate:{:.2f}".format(fps,success_rate), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('result', img)
            # 按下q键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
   