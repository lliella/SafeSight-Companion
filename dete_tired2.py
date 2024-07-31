from PIL import ImageFont, ImageDraw
from tkinter import *
from PIL import Image, ImageTk
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np  # 数据处理的库 numpy
import imutils
import time
import dlib
import cv2
import math
import os
file = 'test1.mp3'
file1 = 'test.mp3'
#参数
#3D参考点
reference_point_3D = np.float32([[6.825897, 6.760612, 4.402142],[1.330353, 7.122144, 6.903745],  
                         [-1.330353, 7.122144, 6.903745],[-6.825897, 6.760612, 4.402142],  
                         [5.311432, 5.485328, 3.987654],[1.789930, 5.393625, 4.413414],  
                         [-1.789930, 5.393625, 4.413414],[-5.311432, 5.485328, 3.987654],  
                         [2.005628, 1.409845, 6.165652],[-2.005628, 1.409845, 6.165652],  
                         [2.774015, -2.080775, 5.048531],[-2.774015, -2.080775, 5.048531],  
                         [0.000000, -3.116408, 6.097667],[0.000000, -7.415691, 4.070434]])  
#添加相机内参
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,0.0, 0.0, 1.0]  
# [fx, 0, cx; 0, fy, cy; 0, 0, 1]
#相机畸变参数[k1, k2, p1, p2, k3]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
#凸轮的本征和畸变系数
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
# 重新投影3D点的世界坐标轴以验证结果姿势
reprojectsrc = np.float32([[10.0, 10.0, 10.0],[10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],[10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],[-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],[-10.0, -10.0, 10.0]])
#绘制正方体12轴
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]
EYE_AR_THRESH = 0.2# EAR阈值
EYE_AR_CONSEC_FRAMES = 3  #当EAR小于阈值时，接连多少帧发生眨眼动作
MAR_THRESH = 0.5
MOUTH_AR_CONSEC_FRAMES = 3
HAR_THRESH = 0.3
NOD_AR_CONSEC_FRAMES = 3
COUNTER = 0# 初始化帧计数器和眨眼总数
TOTAL = 0;eTOTAL = 0;mCOUNTER = 0;mTOTAL = 0;hCOUNTER = 0;hTOTAL = 0;flag = 0
#初始化计时器和计数器
start_time = time.time()
#设置连续闭眼的时间阈值，单位为s,以防误判
PERCLOS_THRESHOLD = 0.15
#检测器
#初始化DLIB的人脸检测器（HOG），然后创建面部标志物预测
#第一步：使用dlib.get_frontal_face_detector() 获得脸部位置检测器
detector = dlib.get_frontal_face_detector()# 人脸检测器
#第二步：使用dlib.shape_predictor获得脸部特征位置检测器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')#人脸68特征点检测
#第三步：分别获取左右眼面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
Pitch_g = 0;Yaw_g = 0;Roll_g = 0
EAR = ''# 返回眼睛的长宽比
MAR = ''
perclos_value=''
color = (0, 255, 255)
eyes_status = []
# 定义计算PERCLOS值的函数
def perclos(eyes_status):
    closed_time = sum(eyes_status)
    total_time = len(eyes_status)
    perclos_value = closed_time / total_time
    print('perclos:',perclos_value )
    return perclos_value

def get_head_pose(shape):  # 头部姿态估计
    # 填写2D参考点
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])
    # solvePnP单目相对位姿估计函数计算姿势获得相机相对于世界坐标系的位姿。——求解旋转和平移矩阵：
    _, rotation_vec, translation_vec = cv2.solvePnP(reference_point_3D, image_pts, cam_matrix, dist_coeffs)
    # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示
    # 计算欧拉角
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接
    # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    global Pitch_g, Yaw_g, Roll_g
    Pitch_g, Yaw_g, Roll_g = pitch, yaw, roll
    return reprojectdst, euler_angle  # 投影误差，欧拉角

def eye_aspect_ratio(eye):#P10图5
    # 垂直眼标志（X，Y）坐标
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):  # 嘴部
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar



def dete_tired(frame):#frame，表示一帧图像
    global EAR, MAR, perclos_value,flag
    frame = imutils.resize(frame, width=660)#将图像缩放到宽度为660像素
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#从BGR颜色空间转换为灰度空间。
    Maximg = np.max(gray)
    Minimg = np.min(gray)
    # 输出最小灰度级和最大灰度级
    Omin, Omax = 0, 255
    # 求 a, b
    a = float(Omax - Omin) / (Maximg - Minimg)
    b = Omin - a * Minimg
    # 线性变换
    img = a * gray + b
    img = img.astype(np.uint8)
    # cv2.imshow('gray',gray)
    # cv2.imshow('enhance',img)
    rects = detector(img, 0)#在灰度图像中检测人脸，并返回一个矩形列表rects。
    if len(rects) == 0:
        font = ImageFont.truetype('simsun.ttc', 36)  # 加载字体，指定字号
        img_pil = Image.fromarray(frame)  # 将帧转换为PIL图像以添加文字
        draw = ImageDraw.Draw(img_pil)
        text = "没有检测到人脸"
        draw.text((100, 140), text, font=font, fill=(0, 255, 0))  # 添加文字到PIL图像
        # 还原帧
        frame = np.array(img_pil)
    mar = ''#初始化一些变量用于存储眼睛、嘴巴和头部的评分、计数器和阈值。
    ear = ''
    global COUNTER, eTOTAL, TOTAL, mCOUNTER, mTOTAL, hCOUNTER, hTOTAL
    # 循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
    for rect in rects:#遍历每个矩形rect
        shape = predictor(img, rect)
        # 将脸部特征信息转换为数组array的格式
        shape = face_utils.shape_to_np(shape)
        # 提取左眼和右眼坐标
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # 嘴巴坐标
        mouth = shape[mStart:mEnd]
        # 构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)
        leftEyeHull = cv2.convexHull(leftEye)#找到左眼、右眼和嘴巴的凸包，绘制轮廓。
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 192, 202), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 192, 202), 1)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (255, 192, 202), 1)
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 192, 202), 1)#在图像上绘制人脸的矩形框
        # 构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)
        font = ImageFont.truetype('simsun.ttc', 22)  # 加载字体，指定字号
        img_pil = Image.fromarray(frame)  # 将帧转换为PIL图像以添加文字
        draw = ImageDraw.Draw(img_pil)
        text2 = "人脸数量: {}".format(len(rects))
        text3 = "计数器: {}".format(COUNTER)
        text4 = "眼睛张合程度比: {:4.2f}".format(ear)
        draw.text((10, 10), text2, font=font, fill=(190, 0, 0))#添加文字到PIL图像
        draw.text((160, 10), text3, font=font, fill=(190, 0, 0))
        draw.text((310, 10), text4, font=font, fill=(190, 0, 0))  
        frame = np.array(img_pil)
        # 满足条件的，眨眼次数+1
        if ear <= EYE_AR_THRESH:
            COUNTER += 1

        else:
            # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
            if ear >= 0.3 and COUNTER > EYE_AR_CONSEC_FRAMES:
                # 重置眼帧计数器
                eTOTAL += 1
                COUNTER = 0

        global EAR, MAR
        EAR = "{:.2f}".format(ear)
        MAR = "{:.2f}".format(mar)
        #计算张嘴评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示打了一次哈欠，同一次哈欠大约在3帧
        if mar > MAR_THRESH:  # 张嘴阈值0.5
            mCOUNTER += 1
            font = ImageFont.truetype('simsun.ttc', 22)  # 加载字体，指定字号
            img_pil = Image.fromarray(frame)  # 将帧转换为PIL图像以添加文字
            draw = ImageDraw.Draw(img_pil)
            text = "打哈欠中"
            draw.text((10, 40), text, font=font, fill=(255, 215, 0))  # 添加文字到PIL图像
            # 还原帧
            frame = np.array(img_pil)
        else:
            # 如果连续3次都大于阈值，则表示打了一次哈欠
            if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:  # 阈值：3
                mTOTAL += 1
            # 重置嘴帧计数器
            mCOUNTER = 0

        #瞌睡点头
        # 获取头部姿态
        reprojectdst, euler_angle = get_head_pose(shape)
        #reprojectdst是一个包含四个点坐标的列表，表示人脸在三维空间中的位置
        har = euler_angle[0, 0]  # 取pitch旋转角度，头部绕x轴旋转的角度
        if mar < 0.3 and har > HAR_THRESH:  # 点头阈值0.3
            hCOUNTER += 1
        else:
            # 如果连续3次都小于阈值，则表示瞌睡点头一次
            if hCOUNTER >= NOD_AR_CONSEC_FRAMES:  #阈值：3
                hTOTAL += 1
            #重置点头帧计数器
            hCOUNTER = 0
        font = ImageFont.truetype('simsun.ttc', 22)  #加载字体，指定字号
        img_pil1 = Image.fromarray(frame)  #将帧转换为PIL图像以添加文字
        draw = ImageDraw.Draw(img_pil1)
        text5 = "俯仰角: {:4.2f}".format(euler_angle[0, 0])
        text6 = "偏航角: {:4.2f}".format(euler_angle[1, 0])
        text7 = "翻滚角:  {:4.2f}".format(euler_angle[2, 0])
        text8 = "计数器: {}".format(mCOUNTER)
        text9 = "嘴巴张合程度比: {:4.2f}".format(mar)
        draw.text((10, 70), text5, font=font, fill=(190, 0, 0))  #添加文字到PIL图像
        draw.text((160, 70), text6, font=font, fill=(190, 0, 0))  
        draw.text((310, 70), text7, font=font, fill=(190, 0, 0))  
        draw.text((160, 40), text8, font=font, fill=(190, 0, 0))  
        draw.text((310, 40), text9, font=font, fill=(190, 0, 0))  
        #还原帧
        frame = np.array(img_pil1)
        for (x, y) in shape:#上画出了人脸特征点的位置。
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        if ear > EYE_AR_THRESH: # 判断眼睛是否闭合
            eyes_status.append(0)  #睁开状态
        else:
            eyes_status.append(1)  #闭合状态
        # 计算PERCLOS值
        global perclos_value
        print('检测帧数', len(eyes_status))
        if len(eyes_status) == 30:
            perclos_value = perclos(eyes_status)
            eyes_status.clear()
            if perclos_value >= PERCLOS_THRESHOLD:
                TOTAL += 1

    #疲劳提示
    if TOTAL >= 5 and mTOTAL >= 2 and hTOTAL >= 3:
        font = ImageFont.truetype('simsun.ttc', 36)  # 加载字体，指定字号
        img_pil = Image.fromarray(frame)  # 将帧转换为PIL图像以添加文字
        draw = ImageDraw.Draw(img_pil)
        text1 = "你已处于疲劳状态"
        text2 = "你已处于疲劳状态,请注意休息！"
        draw.text((100, 100), text1, font=font, fill=(10, 100, 250))  # 添加文字到PIL图像
        # 还原帧
        frame = np.array(img_pil)
        if flag == 0:
            # os.system(file)
            flag = flag+1
        if TOTAL >= 10 and mTOTAL >= 5 and hTOTAL >= 5:
            draw.text((100, 100), text2, font=font, fill=(0, 0, 255))  # 添加文字到PIL图像
            # 还原帧
            frame = np.array(img_pil)
            if flag >= 0:
                # os.system(file1)
                flag = -100
            
    return frame

def take_snapshot():#重置三个全局变量TOTAL
    # video_loop()
    global TOTAL,eTOTAL, mTOTAL, hTOTAL,mCOUNTER,hCOUNTER,COUNTER
    TOTAL = 0;eTOTAL = 0;mTOTAL = 0;hTOTAL = 0;mCOUNTER = 0;hCOUNTER = 0;COUNTER = 0

def video_loop():#
    success, img = camera.read()  # 从摄像头读取照片
    if success:
        detect_result = dete_tired(img)
        cv2image = cv2.cvtColor(detect_result, cv2.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
        current_image = Image.fromarray(cv2image)#将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
        #updata
        global hCOUNTER, mCOUNTER, TOTAL, mTOTAL, hTOTAL
        global Pitch_g, Yaw_g, Roll_g
        global EAR, MAR, perclos_value
        #进行画图操作，68个特征点标识
        root.update()  # 不断更新
        root.after(10)
        Label(root, text='眨眼次数:' + str(eTOTAL), font=("黑体", 14), fg="black", width=16, height=2).place(x=30, y=570, anchor='nw')
        Label(root, text='哈欠次数:' + str(mTOTAL), font=("黑体", 14), fg="black", width=16, height=2).place(x=240, y=570, anchor='nw')
        Label(root, text='点头次数:' + str(hTOTAL), font=("黑体", 14), fg="black", width=16, height=2).place(x=480, y=570, anchor='nw')
        Label(root, text='嘴巴张合程度比: ' + str(MAR), font=("黑体", 14), fg="black", width=20, height=2).place(x=100, y=610, anchor='nw')
        Label(root, text='眼睛张合程度比: ' + str(EAR), font=("黑体", 14), fg="black", width=20, height=2).place(x=350, y=610, anchor='nw')
        root.after(1, video_loop)

if __name__=='__main__':
    
    camera = cv2.VideoCapture(0)  # 摄像头0，笔记本摄像头
    fps = camera.get(cv2.CAP_PROP_FPS)
    # 打印帧率
    print("帧率:", fps)
    root = Tk()
    root.title("opencv + tkinter")
    panel = Label(root)  # initialize image panel
    panel.pack(padx=10, pady=10)
    root.config(cursor="arrow")
    btn = Button(root, text="疲劳提醒解锁", command=take_snapshot)
    btn.pack(fill="both", expand=True, padx=10, pady=10)
    Label(root, text=' ', font=("黑体", 14), fg="red", width=12, height=2).pack(fill="both", expand=True, padx=10, pady=20)
    video_loop()
    root.mainloop()
    # 当一切都完成后，关闭摄像头并释放所占资源
    camera.release()
    cv2.destroyAllWindows()