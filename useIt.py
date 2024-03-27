import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

#建立网络结构:定义了一个卷积神经网络（Convolutional Neural Network，CNN），用于情绪识别的模型。
emotion_model = Sequential()#创建一个序贯模型，用于按顺序堆叠神经网络层。
'''
神经网络层的堆叠:
    Sequential(): 
        序贯模型，与函数式模型对立。
        from keras.models import Sequential， 序贯模型通过一层层神经网络连接构建深度神经网络。
    add(): 
        叠加网络层，参数可为conv2D卷积神经网络层，MaxPooling2D二维最大池化层，
        Dropout随机失活层（防止过拟合），Dense密集层（全连接FC层，在Keras层中FC层被写作Dense层）。
'''
#两个卷积层（Conv2D）：
    # 使用 ReLU 激活函数，分别包含 32 和 64 个卷积核，每个卷积核的大小为 (3, 3)。
    # 第一个卷积层有输入形状 (48, 48, 1)，表示输入图像尺寸为 48x48 的灰度图像。
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#两个最大池化层（MaxPooling2D）：池化窗口大小为 (2, 2)。
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
# 两个 Dropout 层：用于防止过拟合，丢弃率为 0.25。
emotion_model.add(Dropout(0.25))
#再次堆叠两个卷积层和一个最大池化层，卷积核数量分别为 128 和 128，池化窗口大小为 (2, 2)。
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
#一个 Flatten 层：用于将多维输入展平为一维。
emotion_model.add(Flatten())
#一个全连接层（Dense）：包含 1024 个神经元，使用 ReLU 激活函数。
emotion_model.add(Dense(1024, activation='relu'))
#一个 Dropout 层：丢弃率为 0.5。
emotion_model.add(Dropout(0.5))
#最后一个全连接层：包含 7 个神经元，对应于情绪类别数量，使用 softmax 激活函数输出概率分布。
emotion_model.add(Dense(7, activation='softmax'))


emotion_model.load_weights('emotion_model.h5')
'''
`cv2.ocl.setUseOpenCL(False)` 是 OpenCV 中用于禁用 OpenCL 加速的函数调用。
OpenCL（Open Computing Language）是一种用于并行计算的开放式标准，可以利用多核 CPU 和 GPU 的计算能力来加速图像处理和计算任务。
调用 `cv2.ocl.setUseOpenCL(False)` 将禁用 OpenCL 加速，
这意味着 OpenCV 不会再尝试使用 OpenCL 来执行图像处理任务，而会使用传统的 CPU 计算。
有时候在某些环境下，OpenCL 加速可能会导致一些问题，因此禁用 OpenCL 可能有助于解决这些问题。
如果你在使用 OpenCV 进行图像处理时遇到性能或稳定性问题，可以尝试禁用 OpenCL 加速来查看是否有所改善。
'''
cv2.ocl.setUseOpenCL(False)#关闭OpenGL加速。
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


#通过摄像头捕获视频流，并在检测到人脸后识别人脸区域的情绪，并在人脸周围绘制边界框和显示情绪标签。
cap = cv2.VideoCapture(0)# 启动摄像头并捕获视频流，0 表示默认摄像头。通过 cap.read() 读取视频帧。
while True:
    ret, frame = cap.read()#读取视频流的一帧
    if not ret:#ret是 cap.read() 方法的返回值，用于指示是否成功读取到了视频帧。如果成功读取到视频帧，则 ret 的值为 True；如果未成功读取到视频帧，则 ret 的值为 False。
        break
    # 使用 Haar 级联分类器来检测人脸。在每一帧中，使用 detectMultiScale() 方法检测人脸，并返回人脸边界框的坐标。
    bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#加载人脸harr级联分类器。
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#将彩色帧转换为灰度图像。
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)#使用 Haar 级联分类器检测灰度图像中的人脸区域

    for (x, y, w, h) in num_faces:#遍历每个检测到的人脸区域的边界框坐标
        roi_gray_frame = gray_frame[y:y + h, x:x + w]#将灰度图像 gray_frame 中的人脸区域提取出来，存储在 roi_gray_frame 中
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)#裁剪人脸区域并将其调整为模型所需的输入尺寸。
        # 对每个检测到的人脸区域进行情绪识别。通过模型 emotion_model 预测人脸区域的情绪，并找到预测结果中概率最大的情绪类别。
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))#找到情绪预测结果中概率最高的情绪类别索引。
        #绘制边界框和情绪标签
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)# 在图像中的人脸区域周围绘制蓝色矩形边界框
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)#在图像中的人脸区域上方显示识别到的情绪标签。
    #显示视频流：将处理后的视频帧显示在名为 "Video" 的窗口中。
    cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):#当按下 'q' 键时退出视频流捕获。
        break
#释放摄像头资源，并关闭所有 OpenCV 窗口。
cap.release()
cv2.destroyAllWindows()