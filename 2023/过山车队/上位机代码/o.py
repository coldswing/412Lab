import cv2
import numpy as np
import time


#设置色域
th = 100
#my_group = 'b'
low_hsv_red1 = np.array([170, 100, 100])
high_hsv_red1 = np.array([190, 256, 256])
low_hsv_red2 = np.array([0, 100, 100])
high_hsv_red2 = np.array([10, 256, 256])
low_hsv_g = np.array([40, 50, 50])
high_hsv_g = np.array([80, 255, 255])
low_hsv_y = np.array([15, 50, 50])
high_hsv_y = np.array([35, 255, 255])
low_hsv_b = np.array([95, 100, 100])
high_hsv_b = np.array([115, 255, 255])


def baozang_shibie(my_group):
    global low_hsv_red1, high_hsv_red1, low_hsv_red2, high_hsv_red2
    global low_hsv_g, high_hsv_g
    global low_hsv_y, high_hsv_y
    global low_hsv_b, high_hsv_b
    kenel = np.ones((7, 7), np.uint8)
    kenel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cap = cv2.VideoCapture(0) # 调用摄像头‘0’一般q是打开电脑自带摄像头，‘1’是打开外部摄像头（只有一个摄像头的情况）
    width = 300
    height = 300
    # camera = PiCamera()
    # camera.awb_mode = 'off'
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置图像宽度
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # 设置图像高度
    # 显示图像
    while True:

        ret, frame = cap.read()  # 读取图像(frame就是读取的视频帧，对frame处理就是对整个视频的处理)
        # print(ret)#
        #######例如将图像灰度化处理，
        frame = cv2.resize(frame, (400, 300))
        #cv2.imshow('frame',frame)
        img = cv2.GaussianBlur(frame, (3, 3), 0)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转灰度图
        if my_group == 'r':
            # 提取红色区域

            mask_r1 = cv2.inRange(img, lowerb=low_hsv_red1, upperb=high_hsv_red1)
            mask_r2 = cv2.inRange(img, low_hsv_red2, high_hsv_red2)
            mask_r = cv2.morphologyEx(mask_r1 + mask_r2, cv2.MORPH_CLOSE, kenel, iterations=1)
            mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, kenel, iterations=2)

            contours = cv2.findContours(mask_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours[0]]
            max_rect = 0

            if (bounding_boxes):

                for box in bounding_boxes:
                    rect_now = box[2] * box[3]
                    if rect_now > max_rect:
                        max_box = box
                        max_rect = rect_now
                    # if box[2]/(box[1]+1)>2 and box[2]>25:
                cv2.rectangle(frame, (max_box[0], max_box[1]), (max_box[0] + max_box[2], max_box[1] + max_box[3]),
                              (0, 255, 0), 2)
                # else:
                #   bounding_boxes.remove(box)

                # 提取绿色区域
                if max_box[3] > 30 and max_box[3] / max_box[2] > 0.7:

                    mask_g = cv2.inRange(img, low_hsv_g, high_hsv_g)
                    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, kenel1, iterations=1)
                    contours = cv2.findContours(mask_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    bounding_boxes_g = [cv2.boundingRect(cnt) for cnt in contours[0]]

                    cv2.imshow("g", mask_g)

                    mask_y = cv2.inRange(img, low_hsv_y, high_hsv_y)
                    mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, kenel1, iterations=3)
                    contours = cv2.findContours(mask_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    bounding_boxes_y = [cv2.boundingRect(cnt) for cnt in contours[0]]

                    if (bounding_boxes_g):
                        for box1 in bounding_boxes_g:
                            if (box1[0] > max_box[0] and box1[0] + box1[2] < max_box[2] + max_box[0]):
                                if (box1[1] > max_box[1] and box1[3] + box[1] < max_box[3] + max_box[1]):
                                    cv2.rectangle(frame, (box1[0], box1[1]), (box1[0] + box1[2], box1[1] + box1[3]),
                                                  (0, 0, 255), 2)
                                    print("TRUE")
                                    return True
                    elif bounding_boxes_y:
                        for box1 in bounding_boxes_y:
                            if max_box[2] / 4 < box1[2]:
                                cv2.imshow('y', mask_y)
                                if (box1[0] > max_box[0] and (box1[0] + box1[2]) < (max_box[2] + max_box[0])):
                                    if (box1[1] > max_box[1] and (box1[3] + box[1]) < (max_box[3] + max_box[1])):
                                        cv2.rectangle(frame, (box1[0], box1[1]), (box1[0] + box1[2], box1[1] + box1[3]),
                                                      (0, 255, 255), 2)
                                        print("False")
                                        return False

                else:
                    mask_b = cv2.inRange(img, low_hsv_b, high_hsv_b)
                    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, kenel, iterations=1);

                    contours = cv2.findContours(mask_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours[0]]
                    #cv2.imshow("blue", mask_b)
                    if bounding_boxes:
                        for box in bounding_boxes:
                            if box[3] > 30 and box[3] / box[2] > 0.7:
                                print("b_False")
                                return False
                                #cv2.imshow("blue", mask_b)
            else:
                mask_b = cv2.inRange(img, low_hsv_b, high_hsv_b)
                mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, kenel, iterations=1);

                contours = cv2.findContours(mask_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours[0]]
                #cv2.imshow("blue", mask_b)
                if bounding_boxes:
                    for box in bounding_boxes:
                        if box[3] > 30 and box[3] / box[2] > 0.7:
                            print("b_False")
                            return False
                            #cv2.imshow("blue", mask_b)

        else:
            mask_b = cv2.inRange(img, low_hsv_b, high_hsv_b)
            mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, kenel, iterations=1);

           #cv2.imshow('mask_b',mask_b)
            contours = cv2.findContours(mask_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours[0]]
            max_rect = 0

            if (bounding_boxes):

                for box in bounding_boxes:
                    rect_now = box[2] * box[3]
                    if rect_now > max_rect:
                        max_box = box
                        max_rect = rect_now
                    # if box[2]/(box[1]+1)>2 and box[2]>25:
                cv2.rectangle(frame, (max_box[0], max_box[1]), (max_box[0] + max_box[2], max_box[1] + max_box[3]),
                              (0, 255, 0), 2)
                # else:
                #   bounding_boxes.remove(box)

                # 提取绿色区域
                if max_box[3] > 30 and max_box[3] / max_box[2] > 0.7:

                    mask_g = cv2.inRange(img, low_hsv_g, high_hsv_g)
                    mask_g = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, kenel1, iterations=1);
                    contours = cv2.findContours(mask_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    bounding_boxes_g = [cv2.boundingRect(cnt) for cnt in contours[0]]
                    #cv2.imshow("mask_g",mask_g)

                    mask_y = cv2.inRange(img, low_hsv_y, high_hsv_y)
                    mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, kenel1, iterations=3);
                    contours = cv2.findContours(mask_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    bounding_boxes_y = [cv2.boundingRect(cnt) for cnt in contours[0]]
                   #cv2.imshow("mask_y",mask_y)
                    if (bounding_boxes_g):
                        for box1 in bounding_boxes_g:
                            if (box1[0] > max_box[0] and box1[0] + box1[2] < max_box[2] + max_box[0]):
                                if (box1[1] > max_box[1] and box1[3] + box[1] < max_box[3] + max_box[1]):
                                    cv2.rectangle(frame, (box1[0], box1[1]), (box1[0] + box1[2], box1[1] + box1[3]),
                                                  (0, 0, 255), 2)
                                    print("False")
                                    return False
                    elif bounding_boxes_y:
                        for box1 in bounding_boxes_y:
                            if max_box[2] / 4 < box1[2]:
                               #cv2.imshow('y', mask_y)
                                if (box1[0] > max_box[0] and (box1[0] + box1[2]) < (max_box[2] + max_box[0])):
                                    if (box1[1] > max_box[1] and (box1[3] + box[1]) < (max_box[3] + max_box[1])):
                                        cv2.rectangle(frame, (box1[0], box1[1]), (box1[0] + box1[2], box1[1] + box1[3]),
                                                      (0, 255, 255), 2)
                                        print("True")
                                        return True

                else:
                    mask_r1 = cv2.inRange(img, lowerb=low_hsv_red1, upperb=high_hsv_red1)
                    mask_r2 = cv2.inRange(img, low_hsv_red2, high_hsv_red2)
                    mask_r = cv2.morphologyEx(mask_r1 + mask_r2, cv2.MORPH_CLOSE, kenel, iterations=1);
                    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, kenel, iterations=2);

                    contours = cv2.findContours(mask_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours[0]]
                   #cv2.imshow("red", mask_r)
                    if bounding_boxes:
                        for box in bounding_boxes:
                            if box[3] > 40 and box[3] / box[2] > 1:
                                print("r_False")
                                return False
                                #cv2.imshow("blue", mask_b)
            else:
                mask_r1 = cv2.inRange(img, lowerb=low_hsv_red1, upperb=high_hsv_red1)
                mask_r2 = cv2.inRange(img, low_hsv_red2, high_hsv_red2)
                mask_r = cv2.morphologyEx(mask_r1 + mask_r2, cv2.MORPH_CLOSE, kenel, iterations=1);
                mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, kenel, iterations=2);

                contours = cv2.findContours(mask_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours[0]]
               #cv2.imshow("red", mask_r)
                if bounding_boxes:
                    for box in bounding_boxes:
                        if box[3] > 80 and box[3] / box[2] > 1.2:
                            print("r_False")
                            return False
                            #cv2.imshow("blue", mask_b)

        #mask_g = cv2.inRange(img, lowerb=low_hsv_red1, upperb=high_hsv_red1)
        #cv2.imshow('mask_g', mask_g + mask_r)
        #cv2.imshow('ff', mask_r)
        #cv2.imshow('img', img)

        #key = cv2.waitKey(10) & 0xFF
       #if key == ord('q'):
        #    break
    # 释放摄像头并关闭所有窗口
    cap.release()
    # cv2.destroyAllWindows()
