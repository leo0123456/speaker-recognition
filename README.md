# speaker-recognition
语音识别实践课程项目—说话人识别
说话人识别是在通过说话人的语音进行说话人的识别，是将测试需要识别的语音与对应库中的说话人语音模型进行匹配的一个过程。

此次设计基于GMM模型进行说话人识别即声纹识别。

具体实现为首先对说话人的声音进行MFCC提取作为观察向量，利用GMM算法进行说话人的模型训练和测试识别。项目提供友好的人机交互GUI界面，更好的观测感受到识别的效率。经过测试该项目说话人识别准确率较高。
