# emoji_creator_project
学习与实现 https://data-flair.training/blogs/create-emoji-with-deep-learning/  
![4927700bde24b512c282cda085a3d02](https://github.com/SweetThreee/emoji_creator_project/assets/107618206/94c332ab-6af2-435d-85e3-159c199764df)  
## 一、目录结构  
![image](https://github.com/SweetThreee/emoji_creator_project/assets/107618206/209e9b3d-9fe7-48ad-8b35-6fa9f5a4104e)  
## 二、文件说明  
    1.data：存储测试集和验证集的图片（下载链接：https://www.kaggle.com/msambare/fer2013?）  
    2.emojis：存储了七张图片对应七个情绪  
    3.save：存储图形界面中通过保存按钮保存的图片  
    4.train.py：训练模型，产生emotion_model.h5文件  
    5.useIt.py：调用摄像头，展示人脸识别的边框和情绪标签
    6.gui.py：展示图形界面
    7.haarcascade_frontalface_default.xml：人脸harr级联分类器  
## 三、问题  
- **Tensorflow2.9版本安装(python版本3.10)**： https://juejin.cn/post/7160662739524780069
 pip install tensorflow==2.9 -i https://pypi.tuna.tsinghua.edu.cn/simple/ --upgrade tensorflow
- **数据生成器**：train_datagen = ImageDataGenerator(rescale=1./255) 对图像像素值进行缩放，将像素值从 0-255 范围缩放到 0-1 范围  
- **读取并处理图像**：train_generator = train_datagen.flow_from_directory(省略很多参数) 用于从目录中读取图像数据并生成批量的增强图像数据  
- **定义网络结构**：emotion_model = Sequential()创建一个序贯模型，用于按顺序堆叠神经网络层。使用add()添加网络层  
- **编译模型**：emotion_model.compile(): 编译模型，配置损失函数、优化器和评估指标。  
- **训练模型**：emotion_model.fit_generator(): 使用生成器来训练模型，返回一个history对象,记录训练过程中的损失和评估值。  
- **保存模型权重**：emotion_model.save_weights('emotion_model.h5')  
- **加载模型权重**：emotion_model.load_weights('emotion_model.h5')
- **使用 Haar 级联分类器来检测人脸**：bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')，检测人脸，并返回边界框的坐标
- **模型预测**：emotion_prediction = emotion_model.predict(cropped_img)  
 maxindex = int(np.argmax(emotion_prediction))找到情绪预测结果中概率最高的情绪类别索引。
- **Dropout层** ：是一种正则化技术，用于防止神经网络中的过拟合。在训练过程中，Dropout 会随机地将一部分神经元的输出置为零，这样可以减少神经元之间的相互依赖，增强模型的泛化能力。在使用 Dropout 时，参数（权重和偏置）本身并不会发生变化。变化的是在训练过程中，每个批次（batch）的数据通过 Dropout 层时，哪些神经元会被随机地“丢弃”。这意味着每个批次的数据都会经过不同的网络结构，从而增加了模型的多样性。需要注意的是，虽然在训练过程中 Dropout 会影响神经元的输出，但在测试或推理阶段，所有神经元都会被使用。因此，在使用 Dropout 时，通常需要对层的输出进行缩放，以补偿在训练过程中丢失的神经元。具体来说，对于使用 Dropout 的概率为 p 的层，在训练过程中，其输出会乘以 (1-p) 以确保期望值保持不变。而在测试或推理阶段，不需要进行缩放。总之，Dropout 不会改变参数本身，而是通过随机地丢弃一部分神经元来增加模型的多样性，从而提高泛化能力。
- **关于视频流的逐帧读取**
- **Dense层**：也称为全连接层或线性层，是神经网络中的一种基本层类型，其作用是实现特征的全连接组合并在网络中传递信息。  
在神经网络中，Dense层主要负责以下几个方面的功能：  
-- **特征变换与组合**：Dense层通过矩阵乘法将输入数据从一个特征空间线性变换到另一个特征空间，可以学习输入数据中的复杂模式。  
-- **逐层抽象**：在多层神经网络中，Dense层有助于将低层次的特征组合成高层次的特征表示，这种层次化的组合方式有助于网络更好地理解和分类数据。  
-- **连接不同类型的层**：在卷积神经网络（CNN）中，Dense层通常用于连接卷积层和全连接层，以实现特征的全连接组合；在循环神经网络（RNN）中，Dense层则用于连接不同时间步长的输出，以实现序列数据的建模。  
-- **参数学习与优化**：在训练神经网络时，Dense层的参数数量通常是最大的，因此优化Dense层的参数对于提高网络的性能和效率至关重要。常见的优化技巧包括使用正则化、集成学习、学习率调整等方法来控制参数的数量和避免过拟合问题。  
## 四、模型结构  
通过`plot_model(emotion_model, to_file='emotion_model.png', show_shapes=True) `可以得到模型结构的图片展示。  
![867f1d33b1c727f2d5c4baa10c6f72b](https://github.com/SweetThreee/emoji_creator_project/assets/107618206/9b7af4bb-fdf7-46fc-8663-aa1df607180f)
