import cv2

path = "F:/study/6.13finalplan/yuan/"
for i in range(67, 101):
    print(str(i)+'.jpg')
    img = cv2.imread(path+str(i)+'.jpg', cv2.IMREAD_GRAYSCALE)
    img5050 = cv2.resize(img, (300, 300))
    cv2.imshow("img", img5050)
    cv2.waitKey(20)
    cv2.imwrite(r'F:\study\6.13finalplan\yuangray\yuan'+str(i)+'.jpg', img5050)
