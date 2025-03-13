# landslides_identification_model_code
论文名称：改进的DeepLabV3+模型用于震后高分遥感影像滑坡识别（Improved DeepLabV3+ model for landslide identification in high-resolution remote sensing images after earthquakes）

使用数据：谷歌地球影像与中国高分6号遥感影像。

模型设计思路：使用迁移学习方法，将ImageNet数据集上训练好的ResNet50模型的主干网络架构及其参数迁移到DeepLabV3+网络模型中的编码结构中；在解码结构中，使用转置卷积来替换基础DeepLabV3+网络结构中的线性插值操作，进行特征图的上采样处理；此外，为更好的融合编码模块提供的低阶语义信息和高阶语义信息，本文使用通道注意力机制将其融合。如下图所示：![小论文改进的deeplabv3+网络结构图](https://github.com/ZhaoTong0203/landslides_identification_model_code/assets/144538919/f1a4d48d-c320-4747-854f-863aa6fc82c9)

数据集下载地址：
  原始数据：
    链接：https://pan.baidu.com/s/1JmyGDRB7vdaOROuV0xE7Uw 
    提取码：chdl 
  深度学习数据：
    链接：https://pan.baidu.com/s/1qxlKveQD7a4DuA0jvIEQGw 
    提取码：chdl 

If you find this repo useful for your research, please consider citing our paper:
赵通，张双成，何晓宁，薛博维，查富康.2024.改进的DeepLabV3+模型用于震后高分遥感影像滑坡识别.遥感学报，28（9）： 2293-2305 DOI： 10.11834/jrs.20243393. 
Zhao T，Zhang S C，He X N，Xue B W and Zha F K. 2024. Improved DeepLabV3+ model for landslide identification in high-resolution remote sensing images after earthquakes. National Remote Sensing Bulletin， 28（9）：2293-2305 DOI： 10.11834/jrs.20243393. 
