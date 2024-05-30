import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18)):
        super(ASPP, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 1*1的卷积核, 2048*32*32  ---  256*32*32
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0])
        # 3*3的卷积核，此时为6的空洞卷积, 2048*32*32  ---  256*32*32
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1])
        # 3*3的卷积核，此时为12的空洞卷积, 2048*32*32  ---  256*32*32
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2])
        # 3*3的卷积核，此时为18的空洞卷积, 2048*32*32  ---  256*32*32
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 自适应平均池化操作，输出的特征图大小为1*1，计算每个区域内像素的平均值来生成一个1x1的特征图作为输出
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)
        x5 = self.pool(x)  # 2048*32*32  ---  2048*1*1
        x5 = self.conv1(x5)  # 2048*1*1  ---  256*1*1
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=True)  # 256*1*1 --- 256*32*32
        # 这里使用插值操作扩充特征图尺寸
        result = torch.cat([x1, x2, x3, x4, x5], dim=1)  # 1280*32*32
        # 将5组特征图结果进行拼接
        return result


class DeepLabV3plusT(nn.Module):
    """
    deeplab v3+模型，处理RGB三通道影像数据
    """
    def __init__(self, num_classes):
        """
        初始化函数
        :param num_classes: 分类数目
        """
        super(DeepLabV3plusT, self).__init__()

        # Encoder编码部分
        resnet = timm.create_model("tv_resnet50", pretrained=True, features_only=True, in_chans=3, output_stride=8)
        self.encoder1 = nn.Sequential(*list(resnet.children())[0:3])
        self.encoder2 = nn.Sequential(*list(resnet.children())[3:5])
        # 将第二层结果作为低阶特征层结果
        self.encoder3 = nn.Sequential(*list(resnet.children())[5:6])
        self.encoder4 = nn.Sequential(*list(resnet.children())[6:7])
        self.encoder5 = nn.Sequential(*list(resnet.children())[7:8])

        self.aspp = ASPP(in_channels=2048)

        self.conv1 = nn.Conv2d(1280, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        self.up_sample1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.up_sample2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.up_sample3 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv5 = nn.Conv2d(128, num_classes, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(num_classes)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder1(x)
        low_feature = self.encoder2(x)
        x = self.encoder3(low_feature)
        x = self.encoder4(x)
        x = self.encoder5(x)

        high_feature = self.aspp(x)
        high_feature = self.conv1(high_feature)
        high_feature = self.bn1(high_feature)
        high_feature = self.relu(high_feature)
        high_feature = self.up_sample1(high_feature)

        low_feature = self.conv2(low_feature)
        low_feature = self.bn2(low_feature)
        low_feature = self.relu(low_feature)

        middle_feature = torch.cat((high_feature, low_feature), dim=1)

        middle_feature = self.conv3(middle_feature)
        middle_feature = self.bn3(middle_feature)
        middle_feature = self.relu(middle_feature)

        middle_feature = self.up_sample2(middle_feature)

        middle_feature = self.conv4(middle_feature)
        middle_feature = self.bn4(middle_feature)
        middle_feature = self.relu(middle_feature)

        middle_feature = self.up_sample3(middle_feature)

        out_feature = self.conv5(middle_feature)
        out_feature = self.bn5(out_feature)
        out_feature = self.Sigmoid(out_feature)

        return out_feature


if __name__ == "__main__":
    model = DeepLabV3plusT(num_classes=1)
    model_base_path = r"../Model_save/deeplabv3+_model_params.pth"
    model.load_state_dict(torch.load(model_base_path))
    summary(model, input_size=(3, 256, 256), device="cpu")
