import pygame
import numpy as np
import heapq
import matplotlib.pyplot as plt
from itertools import permutations
import cv2
import pygame

# 已知点和目标点
src = np.array([(1, 1), (10, 1), (10, 10), (1, 10)], dtype=np.float32)
dst = np.array([(19, 1), (19, 19), (1, 19), (1, 1)], dtype=np.float32)

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


def p2m(x, y):  # 10*10到21*21
    new_x, new_y = apply_affine_transform(np.array([(x, y)]), M)[0]
    new_x, new_y = round(new_x), round(new_y)
    return new_x, new_y


def m2p(x, y):  # 10*10到21*21
    new_x, new_y = apply_inverse_affine_transform(np.array([(x, y)]), M)[0]
    new_x, new_y = round(new_x, 1), round(new_y, 1)
    return new_x, new_y


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
    path_length = 0
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
            path_length = int((len(path) - 1) / 2)
            return path[::-1], path_length

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
    return [], path_length


def precomputation(map1, start, end, mid_points):
    x = mid_points[:]
    x.append(start)
    x.append(end)
    permutations_list = list(permutations(x, 2))
    length_dict = {}
    for pt in x:
        length_dict[pt] = {}
    for pt1, pt2 in permutations_list:
        length_dict[pt1][pt2] = A_star2(map1, pt1, pt2)[1]
    return length_dict


def get_min_path(map1, start, end, mids):
    """
    返回21*21地图坐标路径点和路径长度
    """
    # 穷举法8！=40320
    # 计算1000个路径需要3s，全部计算需要2分钟计算太慢,但是使用路径查询后大大减少了计算量40320组数据在0.2s完成计算获得最优路径
    length_dict = precomputation(map1, start, end, mids)

    permutations_list = list(permutations(mids))
    # print(permutations_list)
    min_path_length = float("inf")
    min_path = None
    all_points = []
    for mid_points in permutations_list:
        mid_points = list(mid_points)
        # print("mid_points",mid_points)
        mid_points.append(end)
        mid_points.insert(0, start)
        # print("mid_points2",mid_points)
        all_length = 0
        for i in range(len(mid_points) - 1):
            if length_dict:  # 如果没有预计算则采用现场计算，很费时
                length = length_dict[mid_points[i]][mid_points[i + 1]]
                # print(length_dict,mid_points[i],[mid_points[i+1]])
            else:
                length = A_star2(map1, mid_points[i], mid_points[i + 1])[1]
            all_length += length
        if all_length < min_path_length:
            min_path_length = all_length
            min_path = mid_points

    for i in range(len(min_path) - 1):
        for j in A_star2(map1, min_path[i], min_path[i + 1])[0][:-1]:
            all_points.append(j)
    all_points.append(min_path[-1])

    return all_points, min_path_length


def path_plan(map1, start, end, mids):
    # start=(1,1)
    # end=(10,10)
    # mids=[(5,5),(6,6)]
    path = get_min_path(map1, p2m(*start), p2m(*end), [p2m(*i) for i in mids])[0]
    all_points = []
    for i in path:
        all_points.append(m2p(*i))
    return all_points


def generate_border(color=(0, 0, 0), line_scale=5):
    list1 = [
        (0, 0, 10, 0),
        (10, 0, 10, 9),
        (0, 1, 1, 1),
        (2, 0, 2, 3),
        (1, 2, 2, 2),
        (2, 3, 3, 3),
        (3, 3, 3, 1),
        (3, 1, 4, 1),
        (5, 0, 5, 1),
        (4, 2, 5, 2),
        (6, 1, 6, 3),
        (6, 2, 7, 2),
        (4, 3, 6, 3),
        (4, 3, 4, 4),
        (3, 4, 6, 4),
        (0, 3, 1, 3),
        (1, 3, 1, 4),
        (1, 4, 2, 4),
        (1, 5, 3, 5),
        (1, 5, 1, 6),
        (1, 6, 2, 6),
        (2, 6, 2, 7),
        (3, 5, 3, 7),
        (4, 5, 5, 5),
        (0, 7, 1, 7),
        (1, 7, 1, 9),
        (1, 8, 2, 8),
        (2, 8, 2, 9),
        (2, 9, 3, 9),
    ]
    for i in list1:
        start_x, start_y, end_x, end_y = i

        start_x2, start_y2, end_x2, end_y2 = 10 - start_x, 10 - start_y, 10 - end_x, 10 - end_y
        # 将坐标转换为整数
        start_x, start_y, end_x, end_y = int(start_x) * 50 + 150, 650 - int(start_y) * 50, int(
            end_x) * 50 + 150, 650 - int(end_y) * 50

        start_x2, start_y2, end_x2, end_y2 = int(start_x2) * 50 + 150, 650 - int(start_y2) * 50, int(
            end_x2) * 50 + 150, 650 - int(end_y2) * 50

        # 绘制直线
        pygame.draw.line(screen, color, (start_x, start_y), (end_x, end_y), line_scale)
        pygame.draw.line(screen, color, (start_x2, start_y2), (end_x2, end_y2), line_scale)


def pixel2pose(pixel_x, pixel_y):
    """将像素坐标中的点转换到10*10坐标中"""
    pose_x = round((pixel_x - 125) / 50)
    pose_y = 11 - round((pixel_y - 125) / 50)
    return pose_x, pose_y


def pose2pixel(pose_x, pose_y):
    """
    函数接受两个参数，分别为机器人的位置坐标pose_x和pose_y，返回值为该位置对应的图像像素坐标pixel_x和pixel_y。
    """
    pixel_x = 125 + 50 * pose_x
    pixel_y = 675 - 50 * pose_y
    return pixel_x, pixel_y


def draw_paths(paths, color=(139, 0, 139), line_scale=3):
    for i in range(len(paths) - 1):
        start_x, start_y = paths[i]
        end_x, end_y = paths[i + 1]

        # start_x2, start_y2, end_x2, end_y2=10-start_x, 10-start_y, 10-end_x, 10-end_y
        # 将坐标转换为整数
        start_x, start_y, = pose2pixel(start_x, start_y)
        end_x, end_y = pose2pixel(end_x, end_y)

        # start_x2, start_y2, end_x2, end_y2=int(start_x2)*50+150, 650-int(start_y2)*50, int(end_x2)*50+150, 650-int(end_y2)*50
        # 绘制直线
        pygame.draw.line(screen, color, (start_x, start_y), (end_x, end_y), line_scale)
        # pygame.draw.line(screen, color,(start_x2, start_y2),  (end_x2, end_y2), line_scale)
    color1 = (255, 0, 0)  # 起始点
    color2 = (255, 0, 255)  # 中间点
    color3 = (0, 0, 255)  # 末尾点
    text1 = font.render("path_length=%d" % ((len(paths) - 1) / 2), True, (0, 0, 0))
    text_rect1 = text1.get_rect()
    text_rect1.centerx = 400
    text_rect1.centery = 50
    # print(text_rect1.centerx,text_rect1.centery )
    screen.blit(text1, text_rect1)
    text1 = font.render("start_point", True, color1)
    text_rect1 = text1.get_rect()
    text_rect1.centerx = 200
    text_rect1.centery = 700
    # print(text_rect1.centerx,text_rect1.centery )
    screen.blit(text1, text_rect1)
    text1 = font.render("end_point", True, color3)
    text_rect1 = text1.get_rect()
    text_rect1.centerx = 200
    text_rect1.centery = 730
    # print(text_rect1.centerx,text_rect1.centery )
    screen.blit(text1, text_rect1)
    text1 = font.render("intermediate_points", True, color2)
    text_rect1 = text1.get_rect()
    text_rect1.centerx = 200
    text_rect1.centery = 760
    # print(text_rect1.centerx,text_rect1.centery )
    screen.blit(text1, text_rect1)


def draw_points(points):
    radius1 = 10  # 起始，末尾
    radius2 = 8  # 中间
    color1 = (255, 0, 0)  # 起始点
    color2 = (255, 0, 255)  # 中间点
    color3 = (0, 0, 255)  # 末尾点

    if len(points) == 0:
        pass

    elif len(points) == 1:
        pygame.draw.circle(screen, color1, pose2pixel(*points[0]), radius1)

    elif len(points) == 2:
        pygame.draw.circle(screen, color1, pose2pixel(*points[0]), radius1)
        pygame.draw.circle(screen, color3, pose2pixel(*points[-1]), radius1)

    else:
        for index, point in enumerate(points[1:-1]):
            pygame.draw.circle(screen, color2, pose2pixel(*point), radius2)

        pygame.draw.circle(screen, color1, pose2pixel(*points[0]), radius1)
        pygame.draw.circle(screen, color3, pose2pixel(*points[-1]), radius1)
    for index, point in enumerate(points):
        text1 = font.render("%d" % (index), True, (0, 0, 255))
        text_rect1 = text1.get_rect()
        text_rect1.centerx = pose2pixel(*points[index])[0] + 20
        text_rect1.centery = pose2pixel(*points[index])[1]
        # print(text_rect1.centerx,text_rect1.centery )
        screen.blit(text1, text_rect1)


if __name__ == "__main__":
    map1 = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
    generate_border()
    pygame.display.flip()
    # 事件循环

    current_points = []

    while True:
        events = pygame.event.get()
        # print(events)
        for event in events:

            if event.type == pygame.QUIT:
                # 关闭窗口
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # 当鼠标左键单击时，打印鼠标坐标
                pos = pygame.mouse.get_pos()

                current_point = pixel2pose(*pos)
                paths = []
                if 1 <= current_point[0] <= 10 and 1 <= current_point[1] <= 10:
                    current_points.append(current_point)
                    print("确定路径点：", current_point)
                # 计算路径
                if len(current_points) >= 3:
                    paths = path_plan(map1, current_points[0], current_points[-1], current_points[1:-1])
                if len(current_points) == 2:
                    paths = path_plan(map1, current_points[0], current_points[1], [])
                # 绘制
                screen.fill(bg_color)
                generate_border()
                draw_points(current_points)  # 绘制点
                draw_paths(paths)  # 绘制路线
                print("规划路径", paths)
                pygame.display.flip()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                # 当鼠标左键单击时，打印鼠标坐标
                pos = pygame.mouse.get_pos()
                # print("鼠标右键坐标：", pixel2pose(*pos))
                print("清空")
                current_points = []
                print(current_points)
                screen.fill(bg_color)
                generate_border()
                pygame.display.flip()
