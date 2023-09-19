from itertools import permutations
import heapq
import numpy as np
from map import get_maze_map_pose
import cv2
import csv


def save_to_csv(data, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for location in data:
            csv_writer.writerow(location)


if __name__ == "__main__":
    maze_location = [(3, 3)]
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            location, img = get_maze_map_pose(frame)
            location = set(location)
            # print(len(location))
            if (len(location) == 8):
                maze_location = location
                break
        else:
            print("无法读取摄像头")
    print(maze_location)
    save_to_csv(maze_location, 'maze_location_data.csv')

    cap.release()
