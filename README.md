# landslides_identification_model_code
论文名称：改进的DeepLabV3+模型用于震后高分遥感影像滑坡识别（Improved DeepLabV3+ model for landslide identification in high-resolution remote sensing images after earthquakes）

使用数据：谷歌地球影像与中国高分6号遥感影像, 数据集下载地址：详见Data/存放数据集.txt

模型设计思路：使用迁移学习方法，将ImageNet数据集上训练好的ResNet50模型的主干网络架构及其参数迁移到DeepLabV3+网络模型中的编码结构中；在解码结构中，使用转置卷积来替换基础DeepLabV3+网络结构中的线性插值操作，进行特征图的上采样处理；此外，为更好的融合编码模块提供的低阶语义信息和高阶语义信息，本文使用通道注意力机制将其融合。如下图所示：![小论文改进的deeplabv3+网络结构图](https://github.com/ZhaoTong0203/landslides_identification_model_code/assets/144538919/f1a4d48d-c320-4747-854f-863aa6fc82c9)

结果：
网络模型	          召回率	        精确率	        F1 Score	        像素准确率	        平均交并比
FCN		            89.48%	      88.82%	       88.56%	           98.78%	            85.20%
U-Net	            94.15%	      84.73%	       87.72%	           98.36%           	82.32%
SegNet	          83.59%	      87.49%	       84.16%		         98.04%	            79.12%
DeepLabV3+	      95.05%	      89.11%		     91.16%	           98.74%	            86.60%
改进的DeepLabV3+	  92.47%	      90.35%	       90.87%	           98.91%           	87.24%

模型下载地址：详见Save_model/训练模型.txt
