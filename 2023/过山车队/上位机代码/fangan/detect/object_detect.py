
import cv2

# 加载分类器
face_cascade = cv2.CascadeClassifier(r'F:\study\6.13finalplan\train_yuan\xml_file\cascade.xml')
#D:\Final\2023guangdian\haar_img\sanjiao\xml_file\cascade.xml
#D:\Final\2023guangdian\haar_img\yuan\xml_file\cascade.xml
# 打开摄像头
scaleFactor=1.05
minNeighbors=3
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
cv2.namedWindow('image')
def onChange(x):
    global scaleFactor,minNeighbors
    scaleFactor = cv2.getTrackbarPos('scaleFactor', 'image')*0.01+1
    minNeighbors = cv2.getTrackbarPos('minNeighbors', 'image')
  
#cv2.createTrackbar('scaleFactor', 'image', 5, 100, onChange)
#cv2.createTrackbar('minNeighbors', 'image', 20, 100, onChange)
while True:
    # 读取一帧图像
    ret, frame = cap.read()

    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors,minSize=(30, 30))
    # 绘制矩形框
    for (x,y,w,h) in faces:
        new_w = int(w * 2)  # 计算放大后的宽度
        new_h = int(h * 2)  # 计算放大后的高度
        new_x = int(x - (new_w - w) / 2)  # 计算新的左上角x坐标
        new_y = int(y - (new_h - h) / 2)  # 计算新的左上角y坐标
        #cropped_img = frame[new_y:new_y+new_h, new_x:new_x+new_w]
        
        
        color_judge_rect=(int(x+0.4*w),int(y-0.3*h),int(0.2*w),int((0.2*h)))
        x2,y2,w2,h2=color_judge_rect
        color_judge_cropped_img = frame[y2:y2+h2, x2:x2+w2]
        b = color_judge_cropped_img[:,:,0]  # 蓝色通道
        g = color_judge_cropped_img[:,:,1]  # 绿色通道
        r = color_judge_cropped_img[:,:,2]  # 红色通道

        mean_b = round(cv2.mean(b)[0])
        mean_g = round(cv2.mean(g)[0])
        mean_r = round(cv2.mean(r)[0])     
        print(mean_b,mean_g,mean_r)
        if mean_b>max(mean_g,mean_r):
            print("蓝色骨牌圆")
            cv2.putText(frame, "blue_", (new_x,new_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
        elif mean_r>max(mean_g,mean_b):
            print("红色骨牌圆")
            cv2.putText(frame, "red_", (new_x,new_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
        else:
            pass
        cv2.rectangle(frame, (x,y), (x+w,y+h), (mean_b,mean_g,mean_r), 2)#绘制中心贴纸识别框
        #cv2.rectangle(frame, (int(x+0.4*w),int(y-0.3*h)), (int(x+0.6*w),int(y-0.1*h)), (0,255,0), 3)#绘制颜色识别框
        cv2.rectangle(frame, (new_x,new_y), (new_x+new_w,new_y+new_h), (mean_b,mean_g,mean_r), 3)
        #cv2.rectangle(cropped_img, (0.4*new_w,0.1*new_y), (new_x+new_w,new_y+new_h), (0,255,0), 3)
    
    # 显示图像
    #cv2.imshow('cropped_img ', color_judge_cropped_img)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
