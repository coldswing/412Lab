
import cv2

# 加载分类器
face_cascade = cv2.CascadeClassifier(r'F:\study\6.13finalplan\img1\xml_file\cascade.xml')#填写模型文件地址

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取一帧图像
    ret, frame = cap.read()

    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2)

    #if len(faces)!=0:
    #    print(1)
    # 绘制矩形框
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)


    # 显示图像
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
