'''
Descripttion: 
https://www.jianshu.com/p/eece750d348b
opencv_createsamples -info pos.txt -vec pos.vec  -num 4200 -w 30 -h 30
（-info 正样本描述文件，-vec 要生成的vec文件，-num要生成的正样本数， -w生成样本的宽，-h生成样本的高）宽和高要和预处理的w,h对应。
opencv_traincascade -data C:\obeject_detetion\newsamples\classfier -vec C:\obeject_detetion\newsamples\pos2_1\pos.vec -bg C:\obeject_detetion\newsamples\neg_1\neg.txt -w 30 -h 30 -precalcValBufSize 16384 -precalcIdxBufSize 16384  -maxFalseAlarmRate 0.1 -minHitRate 0.999 -mode ALL -numStages 10 -maxWeakCount 200 -numPos 2950 -numNeg 5100 
-data 生成的分类器的目录
-vec 上一步4中生成的vec文件目录
-bg 负样本描述文件目录
-numPos 每一阶段使用的正样本数（越多效果越好）
-numNeg 每一阶段使用的负样本数（和正样本数1：3最好，或者1：4）
-w 样本的宽
-h 样本的高
-precalcValBufSize 分配给训练中每阶段中每样本的内存，越大训练越快,但不能太大，不然会报内存不足的错误，因numPos和numNeg太大引起内存不足报错就减小该参数，不报错前提下越大越好
-precalcIdxBufSize 分配给训练中每阶段中每样本的内存，越大训练越快，但不能太大，不然会报内存不足的错误，因numPos和numNeg太大引起内存不足报错就减小该参数，不报错前提下越大越好，两个加起来不能超过电脑内存
-numStages 训练的阶段数，默认20，太多会过拟合，一般13左右就差不多了
-maxFalseAlarmRate 每一阶段负样本被误判的最大错误率，越小越好，不过训练越久
-minHitRate 每一阶段正样本被正确识别的最小正确率，越大越好，不过训练越久，默认0.995，一般不用设置，设置后-numpos不能设置为全部正样本数，会报数说不正样本数不够。。
-mode 训练使用的特征，默认BASIC只使用垂直的特征，ALL使用垂直和旋转45度的特征，效果更好，不过训练越久。

这是一段训练分类器的Python代码,主要是针对目标检测的应用。这段代码使用OpenCV库来训练分类器,训练的目的是目标检测。
首先,该代码通过输入文件夹路径来指定原始图像文件夹路径、生成的posdata和negdata文件夹路径、xml文件夹路径等。然后,通过PIL库调整pos图像的大小并将其转换为灰度图像,最后保存在指定的文件夹中。同样地,将neg图像调整为指定的大小并转换为灰度图像,然后保存在相应的文件夹中。
接下来,该代码生成posdata.txt和negdata.txt文件,以用于创建正负样本向量。然后,使用OpenCV库中的opencv_createsamples.exe程序生成一个名为detect_number.vec的向量,其中包含了所有的正样本和负样本。最后,使用opencv_traincascade.exe程序训练分类器,并将结果保存在xml文件夹中。
version: 1.0

LastEditTime: 2023-04-11 12:28:47
需要准备一个文件夹,形式如下
        folder
            |
     ---------------
     |             |
    pos           neg
     |             |
  -----         -----
  |   |         |   |
 pos_imgs        neg_imgs

'''
import os
from PIL import Image
import subprocess


pos_size = (50, 50)
neg_size=(500,500)
pos_count=0
neg_count=0
train_step=20
opencv_createsamples=r"C:\Users\LENOVO\Desktop\fangan\fangan\opencv341_bin\opencv_createsamples.exe"
opencv_traincascade=r"C:\Users\LENOVO\Desktop\fangan\fangan\opencv341_bin\opencv_traincascade.exe"

def main(folder_path,pos_size=(50, 50),neg_size=(500, 500),train_step=20,auto_train=False):
    pos_count=0
    neg_count=0
    current_dir=folder_path
    orgin_pos_folder=current_dir+"\\pos\\"
    pos_data_folder=current_dir+"\\posdata\\"
    orgin_neg_folder=current_dir+"\\neg\\"
    neg_data_folder=current_dir+"\\negdata\\"
    xml_file_folder=current_dir+"\\xml_file\\"

    os.mkdir(pos_data_folder)
    os.mkdir(neg_data_folder)
    os.mkdir(xml_file_folder)
    print("创建文件夹成功")
    file_names = os.listdir(orgin_pos_folder)
    for i, file_name in enumerate(file_names):
        # 打开图像文件并将其调整为目标大小和灰度模式
        with Image.open(orgin_pos_folder + file_name) as img:
            img = img.resize(pos_size).convert('L')
            # 构造新的文件名并保存到输出文件夹
            pos_count=i+1
            new_file_name = str(i + 1) + '.bmp'
            img.save(pos_data_folder + new_file_name)
    print("pos图片转换成功,共%d张,size=%d,%d"%(pos_count,pos_size[0],pos_size[1]))
    with open(current_dir+"\\posdata.txt",mode="w",encoding="utf-8") as file:
        for i, file_name in enumerate(file_names):
            new_file_name = "posdata\\"+str(i+1) + '.bmp'

            file.write(new_file_name+" 1 0 0 %d %d"%(pos_size[0],pos_size[1]))
            file.write("\n") 
    print("posdata.txt生成成功")

    file_names = os.listdir(orgin_neg_folder)
    for i, file_name in enumerate(file_names):
        # 打开图像文件并将其调整为目标大小和灰度模式
        with Image.open(orgin_neg_folder + file_name) as img:
            #img = img.resize(neg_size).convert('L')
            img = img.convert('L')
            # 构造新的文件名并保存到输出文件夹
            new_file_name = str(i + 1) + '.bmp'
            neg_count=i + 1
            img.save(neg_data_folder + new_file_name)
    print("neg图片转换成功,共%d张,size=%d,%d"%(neg_count,neg_size[0],neg_size[1]))
    with open(current_dir+"\\negdata.txt",mode="w",encoding="utf-8") as file:
        for i, file_name in enumerate(file_names):
            new_file_name = str(i+1) + '.bmp'
            file.write(neg_data_folder + new_file_name)
            file.write("\n")
    print("negdata.txt生成成功")
    #D:\Final\2023guangdian\opencv_train\opencv341_bin\opencv_createsamples.exe -info D:\Final\2023guangdian\img_detect\posdata.txt -vec D:\Final\2023guangdian\img_detect\detect_number.vec -bg D:\Final\2023guangdian\img_detect\negdata.txt -num 46 -w 50 -h 50
    #D:\Final\2023guangdian\opencv_train\opencv341_bin\opencv_traincascade.exe -data D:\Final\2023guangdian\img_detect\xml_file -vec D:\Final\2023guangdian\img_detect\detect_number.vec -bg D:\Final\2023guangdian\img_detect\negdata.txt -numPos 41 -numNeg 101 -numStages 20 -featureType HAAR -w 50 -h 50

    cmd1="%s -info %s -vec %s -bg %s -num %d -w %d -h %d"%(opencv_createsamples,current_dir+"\\posdata.txt",current_dir+"\\detect_number.vec",current_dir+"\\negdata.txt",pos_count,pos_size[0],pos_size[1])
    cmd2="%s -data %s -vec %s -bg %s -numPos %d -numNeg %d -numStages %d -featureType HAAR -w %d -h %d"%(opencv_traincascade,current_dir+"\\xml_file",current_dir+"\\detect_number.vec",current_dir+"\\negdata.txt",int(pos_count*0.9),neg_count,train_step,pos_size[0],pos_size[1])
    print("运行以下命令进行训练")
    print(cmd1)
    print(cmd2)
    train_cmd=open(current_dir+"\\train_cmd.txt","w",encoding="utf-8")
    train_cmd.write(cmd1)
    train_cmd.write("\n")
    train_cmd.write(cmd2)
    train_cmd.close()
    
    if auto_train:
        print("开始自动训练")
        #command = r'D:\Final\2023guangdian\opencv_train\opencv341_bin\opencv_traincascade.exe -data D:\Final\2023guangdian\imgtrain\xml_file -vec D:\Final\2023guangdian\imgtrain\detect_number.vec -bg D:\Final\2023guangdian\imgtrain\negdata.txt -numPos 41 -numNeg 101 -numStages 20 -featureType HAAR -w 50 -h 50'
        process = subprocess.Popen(cmd1.split(), stdout=subprocess.PIPE)

        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                print(output.strip().decode())

        if process.returncode == 0:
            print('Command executed successfully')
        else:
            print(f'Command failed with return code {process.returncode}')
        process = subprocess.Popen(cmd2.split(), stdout=subprocess.PIPE)

        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                print(output.strip().decode())

        if process.returncode == 0:
            print('Command executed successfully')
        else:
            print(f'Command failed with return code {process.returncode}')


        print("生成分类器保存在%s"%(xml_file_folder+"newyuan.xml"))

main(r"F:\study\6.13finalplan\train_yuan",auto_train=True,train_step=15)
#D:\Final\2023guangdian\opencv_train\opencv341_bin\opencv_createsamples.exe -info E:\edgeDownloads\face_Data\Train\posdata.txt -vec E:\edgeDownloads\face_Data\Train\detect_number.vec -bg E:\edgeDownloads\face_Data\Train\negdata.txt -num 5000 -w 50 -h 50
#D:\Final\2023guangdian\opencv_train\opencv341_bin\opencv_traincascade.exe -data E:\edgeDownloads\face_Data\Train\xml_file -vec E:\edgeDownloads\face_Data\Train\detect_number.vec -bg E:\edgeDownloads\face_Data\Train\negdata.txt -numPos 4500 -numNeg 5000 -numStages 20 -featureType HAAR -w 50 -h 50