import cv2
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk
from datetime import datetime
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

#鸣谢：https://blog.csdn.net/QQ546475772/article/details/136143667?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-136143667-blog-117086277.235%5Ev43%5Econtrol&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-136143667-blog-117086277.235%5Ev43%5Econtrol&utm_relevant_index=6
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('emotion_model.h5')
cv2.ocl.setUseOpenCL(False)#关闭OpenGL加速。
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
# 显示图片
def show_frame():
    global frame  # 将frame声明为全局变量
    # 读取摄像头画面
    ret, frame = cap.read()
    bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # 加载人脸harr级联分类器。
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将彩色帧转换为灰度图像。
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)  # 使用 Haar 级联分类器检测灰度图像中的人脸区域

    for (x, y, w, h) in num_faces:  # 遍历每个检测到的人脸区域的边界框坐标
        roi_gray_frame = gray_frame[y:y + h, x:x + w]  # 将灰度图像 gray_frame 中的人脸区域提取出来，存储在 roi_gray_frame 中
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1),
                                     0)  # 裁剪人脸区域并将其调整为模型所需的输入尺寸。
        # 对每个检测到的人脸区域进行情绪识别。通过模型 emotion_model 预测人脸区域的情绪，并找到预测结果中概率最大的情绪类别。
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))  # 找到情绪预测结果中概率最高的情绪类别索引。
        # 绘制边界框和情绪标签
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)  # 在图像中的人脸区域周围绘制蓝色矩形边界框
        cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)  # 在图像中的人脸区域上方显示识别到的情绪标签。
        img = Image.open('./emojis/{}.png'.format(emotion_dict[maxindex]))
        # print(img.size)
        img=img.resize((200,200))
        photo = ImageTk.PhotoImage(img)
        global image_Label
        image_Label.config(image=photo)
        image_Label.image=photo
        global label_emoji
        label_emoji.config(text=f"{emotion_dict[maxindex]}")
    # 获取当前时间
    current_time = datetime.now()

    # 将时间转换为字符串
    time_str = current_time.strftime("%Y%m%d %H:%M:%S")

    # 在帧上添加时间文本
    cv2.putText(frame, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 将OpenCV图像转换为Tkinter图像
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将颜色通道顺序转换为RGB
    image = Image.fromarray(frame_rgb)
    image = ImageTk.PhotoImage(image)

    # 更新标签中的图像
    label.configure(image=image)
    label.image = image

    # 每隔10毫秒调用一次show_frame函数
    label.after(10, show_frame)


# 保存图片
def save_pic():
    global image_names
    # 获取当前时间
    current_time = datetime.now()
    # 构建图片名称
    image_name = current_time.strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
    # 显示文字
    image_name_dis = "保存成功: \n" + image_name
    # 构建保存路径
    save_folder = "./save"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, image_name)
    # 保存当前帧为图片
    success = cv2.imwrite(save_path, frame)
    # 将新的图片名称添加到列表中
    image_names.append(image_name_dis)
    # 将所有的图片名称拼接起来，并设置为label_filename标签的文本内容，每个图片名称之间添加换行符
    label_filename.config(text="\n".join(image_names))

# def motion(event):
#     x,y=event.x,event.y
#     label_emoji.config(text="({},{})".format(x,y))
#     print("鼠标位置：x=", x, " y=", y)

if __name__ == "__main__":
    # 声明一个列表来存储所有的图片名称
    image_names = []

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # 创建Tkinter窗口
    window = tk.Tk()
    window.title("情绪识别")
    window.geometry("900x500")  # 设置窗口的宽度和高度为800x600像素
    window['bg'] = 'white'
    # 创建显示视频的标签
    label = tk.Label(window)
    label.place(x=10, y=10, width=640, height=480)  # 设置标签的位置和大小
    # 创建保存图片名称的标签
    label_filename = tk.Label(window,bg="white")
    label_filename.place(x=690, y=300, width=200, height=40)  # 设置标签的位置和大小
    # 创建保存按钮
    button_save = tk.Button(window, text="保存图片",bg="white", command=save_pic,font=('arial', 10, 'bold'))
    button_save.place(x=720, y=400, width=100, height=40)  # 设置按钮的位置和大小
    exitbutton = tk.Button(window, text='Quit',bg="white", command=window.destroy, font=('arial', 10, 'bold'))
    exitbutton.place(x=720, y=450, width=100, height=40)  # 设置按钮的位置和大小
    #结果显示

    img = Image.open('emojis/Angry.png')
    img=img.resize((200,200))
    print(img.size)
    photo = ImageTk.PhotoImage(img)
    image_Label = tk.Label(window, image=photo,bg="white")
    image_Label.place(x=670, y=50, width=200, height=250)
    # image_Label.pack()
    label_emoji = tk.Label(window, text="happy", bg="white", font=('arial', 20, 'bold'))
    label_emoji.place(x=720, y=20, width=100, height=30)  # 设置标签的位置和大小
    # 开始显示视频
    show_frame()

    # 运行Tkinter事件循环
    window.mainloop()
