# emoji_creator_project
学习与实现 https://data-flair.training/blogs/create-emoji-with-deep-learning/  
一、目录结构  
![image](https://github.com/SweetThreee/emoji_creator_project/assets/107618206/209e9b3d-9fe7-48ad-8b35-6fa9f5a4104e)  
二、文件说明  
    1.data：存储测试集和验证集的图片（下载链接：https://www.kaggle.com/msambare/fer2013?）  
    2.emojis：存储了七张图片对应七个情绪  
    3.save：存储图形界面中通过保存按钮保存的图片  
    4.train.py：训练模型，产生emotion_model.h5文件  
    5.useIt.py：调用摄像头，展示人脸识别的边框和情绪标签
    6.gui.py：展示图形界面
    7.haarcascade_frontalface_default.xml：人脸harr级联分类器  
三、模型结构  
通过`plot_model(emotion_model, to_file='emotion_model.png', show_shapes=True) `可以得到模型结构的图片展示。  
![867f1d33b1c727f2d5c4baa10c6f72b](https://github.com/SweetThreee/emoji_creator_project/assets/107618206/9b7af4bb-fdf7-46fc-8663-aa1df607180f)
