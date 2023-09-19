import cv2


def capture_images():
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 创建视频编码器并设置相关参数
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_filename = 'captured_video.mp4'
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

    while True:
        # 读取摄像头的图像帧
        ret, frame = cap.read()

        if not ret:
            print("无法获取图像帧")
            break

        # 显示图像帧
        cv2.imshow('Camera', frame)

        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 将图像帧写入输出文件
        out.write(frame)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("图像捕捉完成，保存为", output_filename)


# 调用函数开始捕捉图像
capture_images()
