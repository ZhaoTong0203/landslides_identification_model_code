# landslides_identification_model_code
论文名称：改进的DeepLabV3+模型用于震后高分遥感影像滑坡识别（Improved DeepLabV3+ model for landslide identification in high-resolution remote sensing images after earthquakes）

使用数据：谷歌地球影像与中国高分6号遥感影像, 数据集下载地址：详见Data/存放数据集.txt（未上传）

模型设计思路：使用迁移学习方法，将ImageNet数据集上训练好的ResNet50模型的主干网络架构及其参数迁移到DeepLabV3+网络模型中的编码结构中；在解码结构中，使用转置卷积来替换基础DeepLabV3+网络结构中的线性插值操作，进行特征图的上采样处理；此外，为更好的融合编码模块提供的低阶语义信息和高阶语义信息，本文使用通道注意力机制将其融合。如下图所示：![小论文改进的deeplabv3+网络结构图](https://github.com/ZhaoTong0203/landslides_identification_model_code/assets/144538919/f1a4d48d-c320-4747-854f-863aa6fc82c9)

完整数据与代码暂未上传，目前仅提供部分数据集样例，下载地址如下：
链接：https://pan.baidu.com/s/1iz0tZVB2D18jlwxWDqdpUg 
提取码：chdl 
