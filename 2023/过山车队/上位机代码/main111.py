"""主要逻辑代码
需要与底层控制通讯，树莓派发送控制指令到动作执行机构
需要完善以下部分 ，参考注释编写代码
car_move
move_with_one_line
attactk
recive_msg
"""

import time
from itertools import permutations
import heapq
import numpy as np
from map import get_maze_map_pose
import cv2
import serial as ser
#最基本的A*搜索算法
def A_star(map, start, end):
    """
    使用A*算法寻找起点到终点的最短路径
    :param map: 二维列表，表示地图。0表示可以通过的点，1表示障碍物。
    :param start: 元组，表示起点坐标。
    :param end: 元组，表示终点坐标。
    :return: 列表，表示从起点到终点的最短路径，其中每个元素是一个坐标元组。
    """
    # 定义启发式函数（曼哈顿距离）
    def heuristic(node1, node2):
        return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

    # 初始化open_list、closed_list、g_score、came_from
    open_list = [(0, start)]
    closed_list = set()
    g_score = {start: 0}
    came_from = {}

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
            return path[::-1]

        # 将当前节点加入closed_list
        closed_list.add(current)

        # 遍历相邻节点
        for neighbor in [(current[0] - 1, current[1]),
                         (current[0] + 1, current[1]),
                         (current[0], current[1] - 1),
                         (current[0], current[1] + 1)]:
            if 0 <= neighbor[0] < len(map) and 0 <= neighbor[1] < len(map[0]) and map[neighbor[0]][neighbor[1]] == 0:
                # 相邻节点是可通过的节点
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # 如果相邻节点不在g_score中，或者新的g值更优，则更新g_score和came_from
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_list, (f_score, neighbor))
                    came_from[neighbor] = current

    # 没有找到可行路径，返回空列表
    return []

#坐标变换函数，将10*10的坐标映射到地图矩阵上，方便用来可视化
def pose2map(x,y):
    return 21-2*y,x*2-1

def map2pose(x,y):
    return (21-y)/2,(x+1)/2

#地图上两点之间的最短路径
def A_star_length(map1,start_x,start_y,end_x,end_y):
    # 定义起点和终点
    start = (start_x, start_y)
    end = (end_x, end_y)

    # 计算最短路径
    start=pose2map(*start)
    end=pose2map(*end)
    path = A_star(map1, start, end)
    path_length=int((len(path)-1)/2)
    return path_length

#预计算
def precomputation(map1,start,end,mid_points):
    permutations_list = list(permutations(mid_points,2))
    length_dict={}
    length_dict[start]={}
    for pt1 in mid_points:
        length_dict[pt1]={}
        length_dict[start][pt1]=A_star_length(map1,start[0],start[1],pt1[0],pt1[1])
    for pt1 in mid_points:
        length_dict[pt1][end]=A_star_length(map1,pt1[0],pt1[1],end[0],end[1])
    for pt1,pt2 in permutations_list:
        length_dict[pt1][pt2]=A_star_length(map1,pt1[0],pt1[1],pt2[0],pt2[1])
    length_dict[start][end]=A_star_length(map1,start[0],start[1],end[0],end[1])
    return length_dict

#计算最短距离的路线
def get_min_path(map1,start,end,mid_points,length_dict=None):
    #穷举法8！=40320
    #计算1000个路径需要3s，全部计算需要2分钟计算太慢,但是使用路径查询后大大减少了计算量40320组数据在0.2s完成计算获得最优路径
    permutations_list = list(permutations(mid_points))
    min_path_length=float("inf")
    min_path=None
    for mid_points in permutations_list:
        mid_points=list(mid_points)
        mid_points.append(end)
        mid_points.insert(0,start)

        all_length=0
        for i in range(len(mid_points)-1):
            if length_dict:#如果没有预计算则采用现场计算，很费时
                length=length_dict[mid_points[i]][mid_points[i+1]]
            else:
                length=A_star_length(map1,mid_points[i][0],mid_points[i][1],mid_points[i+1][0],mid_points[i+1][1])
            all_length+=length
        if all_length<min_path_length:
            min_path_length=all_length
            min_path=mid_points

    return min_path,min_path_length 

#将10*10pose坐标映射到21*21的地图坐标上
def gennerate_all_path(map1,min_path):
    path=[]
    for i in range(len(min_path)-1):
        #start=pose2map(*start)
        #end=pose2map(*end)
        base_path=A_star(map1,pose2map(*min_path[i]),pose2map(*min_path[i+1]))
        path+=base_path[1:]
    path.insert(0,pose2map(*min_path[0]))
    return path

def multi_goal_Astar(map1,start,end,mid_points):
    '''
    含有中间位置的最短路径规划算法
    '''
    yujisuan=precomputation(map1,start,end,mid_points)
    min_path,min_path_length =get_min_path(map1,start,end,mid_points,yujisuan)

    #print(real) 
    
    return min_path,min_path_length

def multi_Astar(map1,start,end,mid_points):
    min_path,min_path_length=multi_goal_Astar(map1,start,end,mid_points)
    all_points=[]
    for i in range(len(min_path)-1):
        temp=A_star(map1,pose2map(*min_path[i]),pose2map(*min_path[i+1]))[:-1]
        for j in temp:
            all_points.append(j)
    all_points.append(pose2map(*min_path[-1]))
    real=[]
    for point in all_points:
        real.append(map2pose(point[0],point[1]))
    #print(all_points)
    real=real[::-1]
    return real

def find_turning_points(path):
    turning_points = []
    for i in range(1, len(path)-1):
        current = path[i]
        previous = path[i-1]
        next = path[i+1]
        if ((current[0]-previous[0]) * (next[1]-current[1]) != (current[1]-previous[1]) * (next[0]-current[0])) or(current[0]-previous[0]) * (next[0]-current[0]) + (current[1]-previous[1]) * (next[1]-current[1]) <0:#or (current[0]-previous[0]) * (next[1]-current[1])==(current[1]-previous[1]) * (next[0]-current[0]) ==0#(3,1) (3,3) (3,1) (3-3)*(1-3) (3-1)*(1-3)
            turning_points.append(current)


    return turning_points

def turn_direction(v1, pt1,pt2):
    x1,y1=pt1
    x2,y2=pt2
    # 计算下一个方向向量
    v2 = np.array([x2, y2]) - np.array([x1, y1])
    # 计算叉积
    cross = np.cross(v1, v2)
    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
    norm_product = ((v1[0]**2 + v1[1]**2) *(v2[0]**2 + v2[1]**2))**0.5
    if cross > 0:
        return 'left'
    elif cross < 0:
        return 'right'
    elif dot_product==norm_product:
        return "straight"
    else:
        return 'Reverse direction'

def car_move(action):
    """控制小车改变运行方向
    action:"right" "left" "straight" "Reverse direction"
    """
    #pass
    #在这里添加控制车辆转向的代码，使用串口通讯向stm32发送指令，必须执行完动作才能返回，阻塞！
    if action == 'right':
        ser.write(b'3')
    if action == 'left':
        ser.write(b'2')
    if action == 'straight':
        ser.write(b'1')
    if action == 'Reverse direction':
        ser.write(b'4')
    return 

def recive_msg():
    """recive_msg 接收stm32发送过来识别到岔路口的信号
    如果识别到岔路口，返回True否则返回False
    在这里添加你的代码，接收消息
    """
    data=ser.read(10)
    if data is not None:
        return True
    return False
        

def attactk():
    """attactk 撞击宝藏
    """
    pass
    #在这里添加碰撞宝藏的代码，不能转动车的方向，建议使用舵机插上小棍，在车前面摆动，执行完动作之后返回，一定要阻塞
    #控制舵机直行动作可以使用树莓派，也可以使用stm32,会产生pwm波就可以
    return

def DO(action_list):
    """

    """
    dict={"left":1,"right":2,"straight":3,'Reverse direction':4}
    arr=[]
    for i in action_list:
        if i in dict:
            arr.append(dict[i])
        else:
            continue
    print("over")
    return arr

def SER(arr):
    ser.write(b'4')
    time.sleep(0.1)
    ser.write(b'1')
    time.sleep(0.1)
    ser.write(b'2')
    time.sleep(0.1)
    for i in arr:
        ser.write(bytes(str(i),'utf-8'))
    ser.write(b'0')



if __name__=="__main__":
# 初始化串口
    #ser=ser.Serial('/dev/ttyTHS1',115200,timeout=0.5)
    #time.sleep(1)

    #step1 识别宝藏
    maze_location=[(3, 3)]
    cap=cv2.VideoCapture(0)
    ret, frame=cap.read()
    while True:
        ret, frame=cap.read()
        if ret:
            location,img=get_maze_map_pose(frame)
            location=set(location)
            #print(len(location))
            if(len(location)==8):
                maze_location=location
                break
        else:
            print("无法读取摄像头")
    #print(maze_location)

    
    
    #step2路径规划
    start=(1, 1)
    end=(10,10)

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
    
    print("开始路径规划")
    min_path=multi_Astar(map1,start,end,maze_location)
    turn_points=min_path
    print("路径规划已完成")
    print(turn_points)
    
    #step3计算岔路口的动作
    now_dir=(1,0)
    #print("设定初始方向",now_dir)

    cross_points=[(2,1),(2,2),(1,2),(1,3),(2,3),(2,4),(3,4),(4,4),(4,3),(4,2),(5,2),(5,1),(3,1),(1,5),(1,7),(1,10),
    (2,7),(2,8),(2,10),(3,5),(3,6),(3,8),(3,9),(4,5),(4,6),(4,7),(4,8),(4,9),(4,10),(5,8),(5,9),(5,10),(6,1),(6,2),
    (6,3),(6,9),(6,10),(7,1),(7,3),(7,4),(7,5),(7,6),(7,7),(7,8),(7,9),(8,2),(8,3),(8,5),(8,6),(8,7),(8,10),(9,1),
    (9,3),(9,4),(9,7),(9,8),(9,9),(9,10),(10,1),(10,4),(10,6),(10,8),(10,9)]

    cross_points.extend(maze_location)

    #print("准备动作执行列表")
    action_list=[]
    for i in range(len(turn_points)-1):
        action=turn_direction(now_dir, turn_points[i], turn_points[i+1])
        now_dir=turn_points[i+1][0]-turn_points[i][0],turn_points[i+1][1]-turn_points[i][1]
        if turn_points[i] in cross_points:# or action=="Reverse direction":
            #print(i,turn_points[i],action)
            action_list.append(action)
    #action_list=[]#遇到多叉路口执行动作的列表"right" "left" "straight" "Reverse direction"
    print(action_list)
    #arr=DO(action_list)
    #SER(arr)
    #step4开始寻线行驶
    maze_last=[]
    for i in range(len(turn_points)-1):
        if turn_points[i] in cross_points:
            maze_last.append(turn_points[i])


    print("开始")

    maze_num=0
    flag2=0
    index=0
    cross_points_last=[]
    if maze_num==3 :
        start = maze_last[index-1]
        maze_location=[]
        min_path = multi_Astar(map1, start, end, maze_location)
        turn_points = min_path
        print("最终规划已完成")
        # print(turn_points)
        now_dir = (1, 0)
        action_list_last = []
        for i in range(len(turn_points) - 1):
            action = turn_direction(now_dir, turn_points[i], turn_points[i + 1])
            now_dir = turn_points[i + 1][0] - turn_points[i][0], turn_points[i + 1][1] - turn_points[i][1]
        if turn_points[i] in cross_points:  # or action=="Reverse direction":
            action_list_last.append(action)
        # arr=DO(action_list)
        # SER(arr)

    #ser.write(b'1')
    A_cascade = cv2.CascadeClassifier(r'F:\study\6.13finalplan\img1\xml_file\sanjiao.xml')
    B_cascade = cv2.CascadeClassifier(r'F:\study\6.13finalplan\img2\xml_file\yuan.xml')
    red1=0 # 蓝三角
    red2=0 # 红三角
    blue1=0 # 蓝圆
    blue2=0 # 红圆
    time.sleep(0.1)

    while True:
        if maze_num == 3:
            start = maze_last[index - 1]
            maze_location = []
            min_path = multi_Astar(map1, start, end, maze_location)
            turn_points = min_path
            print("最终规划已完成")
            # print(turn_points)
            now_dir = (1, 0)
            action_list_last = []
            for i in range(len(turn_points) - 1):
                action = turn_direction(now_dir, turn_points[i], turn_points[i + 1])
                now_dir = turn_points[i + 1][0] - turn_points[i][0], turn_points[i + 1][1] - turn_points[i][1]
            if turn_points[i] in cross_points:  # or action=="Reverse direction":
                action_list_last.append(action)
            # arr=DO(action_list)
            # SER(arr)
            break

        ret, frame = cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        A = A_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
        B = B_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)
        for (x,y,w,h) in A:
            new_w = int(w * 2)  # 计算放大后的宽度
            new_h = int(h * 2)  # 计算放大后的高度
            new_x = int(x - (new_w - w) / 2)  # 计算新的左上角x坐标
            new_y = int(y - (new_h - h) / 2)  # 计算新的左上角y坐标
            color_judge_rect = (int(x + 0.4 * w), int(y - 0.3 * h), int(0.2 * w), int((0.2 * h)))
            x2, y2, w2, h2 = color_judge_rect
            color_judge_cropped_img = frame[y2:y2 + h2, x2:x2 + w2]
            b = color_judge_cropped_img[:, :, 0]  # 蓝色通道
            g = color_judge_cropped_img[:, :, 1]  # 绿色通道
            r = color_judge_cropped_img[:, :, 2]  # 红色通道

            mean_b = round(cv2.mean(b)[0])
            mean_g = round(cv2.mean(g)[0])
            mean_r = round(cv2.mean(r)[0])
            #print(mean_b, mean_g, mean_r)
            if mean_b > max(mean_g, mean_r):
                red1=red1+1
                #print("蓝三角")
            elif mean_r > max(mean_g, mean_b):
                red2=red2+1
                #print("红三角")
            else:
                continue
        for (x,y,w,h) in B:
            new_w = int(w * 2)  # 计算放大后的宽度
            new_h = int(h * 2)  # 计算放大后的高度
            new_x = int(x - (new_w - w) / 2)  # 计算新的左上角x坐标
            new_y = int(y - (new_h - h) / 2)  # 计算新的左上角y坐标
            color_judge_rect = (int(x + 0.4 * w), int(y - 0.3 * h), int(0.2 * w), int((0.2 * h)))
            x2, y2, w2, h2 = color_judge_rect
            color_judge_cropped_img = frame[y2:y2 + h2, x2:x2 + w2]
            b = color_judge_cropped_img[:, :, 0]  # 蓝色通道
            g = color_judge_cropped_img[:, :, 1]  # 绿色通道
            r = color_judge_cropped_img[:, :, 2]  # 红色通道

            mean_b = round(cv2.mean(b)[0])
            mean_g = round(cv2.mean(g)[0])
            mean_r = round(cv2.mean(r)[0])
            #print(mean_b, mean_g, mean_r)
            if mean_b > max(mean_g, mean_r):
                blue1=blue1+1
                #print("蓝圆")
            elif mean_r > max(mean_g, mean_b):
                blue2=blue2+1
                #print("红圆")
            else:
                continue
        if red2 > 3 :
            print("红三角")
            maze_num=maze_num+1
            index=index+1
            red1 = 0
            red2 = 0
            blue1 = 0
            blue2 = 0
        if blue1 > 5 :
            print("蓝圆")
            # ser.write(b'4')#跑路
            index = index + 1
            red1 = 0
            red2 = 0
            blue1 = 0
            blue2 = 0
        if red1 > 8 :
            print("蓝三角")
            #ser.write(b'4')#跑路
            index = index + 1
            red1=0
            red2 = 0
            blue1 = 0
            blue2 = 0
        if blue2 > 5 :
            print("红圆")
            # ser.write(b'4')#跑路
            index = index + 1
            red1 = 0
            red2 = 0
            blue1 = 0
            blue2 = 0
    cap.release()

    #ser.close()
