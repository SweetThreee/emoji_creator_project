from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'data/train'
val_dir = 'data/validation'
'''
ImageDataGenerator：图像数据生成器,用于图像数据增强和预处理
    from keras.preprocessing.image import ImageDataGenerator
    对图片进行批量预处理:
        rescale=1./255 
        表示对图像像素值进行缩放，将像素值从 0-255 范围缩放到 0-1 范围，这是常见的图像预处理步骤之一。
'''
train_datagen = ImageDataGenerator(rescale=1./255)#用于训练集的数据生成器
val_datagen = ImageDataGenerator(rescale=1./255)
'''
flow_from_directory：用于从目录中读取图像数据并生成批量的增强图像数据。
    以文件夹路径为参数，生成经过数据提升/归一化后的数据，参数包括目录名val_dir、批尺寸batch_size等。
'''
train_generator = train_datagen.flow_from_directory(
        train_dir,#指定训练数据所在的目录。
        target_size=(48,48),#将输入图像调整为指定的大小 (48x48)。
        batch_size=64,#每个批次中包含的图像样本数量为 64。
                      #batch_size=total即梯度下降法，bach_size=1即随机梯度下降法SGD；
        color_mode="grayscale",#指定图像的颜色模式为灰度图像（常见取值有 "grayscale"（灰度图像）和 "rgb"（彩色图像）。）
        class_mode='categorical')#指定分类问题的类别类型为分类标签，这表示数据集中的标签将被转换为 one-hot 编码的形式。
                                #常见取值有 "categorical"（one-hot 编码的类别标签）、"binary"（二进制标签）、"sparse"（整数标签）、"input"（不返回标签，仅返回图像数据）等。
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')
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

'''
针对情绪识别模型进行编译和训练:
    emotion_model.compile(): 编译模型，配置损失函数、优化器和评估指标。
        loss='categorical_crossentropy': 使用交叉熵损失函数，适用于多类别分类问题。
        optimizer=Adam(lr=0.0001, decay=1e-6): 使用 Adam 优化器，学习率为 0.0001，学习率衰减为 1e-6。
        metrics=['accuracy']: 设置模型评估指标为准确率。
    emotion_model.fit_generator(): 使用生成器来训练模型，返回一个history对象,记录训练过程中的损失和评估值。
        train_generator: 训练数据的生成器函数。
        steps_per_epoch=28709//64: 每个 epoch 中从生成器中抽取的步数（生成器执行生成数据的次数），通常为样本数除以批量大小。
        epochs=50: 训练的 epoch 数量。
        validation_data=validation_generator: 数据集验证=验证数据的生成器。
        validation_steps=7178//64: 在每个 epoch 结束时从验证生成器中抽取的步数，用于验证模型性能。    
'''
emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=0.0001, decay=1e-6),
    metrics=['accuracy'])
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7178 // 64)
#保存模型权重
emotion_model.save_weights('emotion_model.h5')


