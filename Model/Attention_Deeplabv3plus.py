import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )  # 256*64*64 --- 256*64*64

    def forward(self, x, y):
        attention_x = self.attention(x)
        attention_y = self.attention(y)

        return x * attention_x + y * attention_y


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18)):
        super(ASPP, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0])
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1])
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2])
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)
        x5 = self.pool(x)
        x5 = self.conv1(x5)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=True)
        result = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return result


class AttentionDeeplabV3plus(nn.Module):

    def __init__(self, num_classes):
        """
        初始化函数
        :param num_classes: 分类数目
        """
        super(AttentionDeeplabV3plus, self).__init__()

        # Encoder编码部分
        resnet = timm.create_model("tv_resnet50", pretrained=True, features_only=True, in_chans=3, output_stride=8)

        self.encoder1 = nn.Sequential(
            nn.Sequential(*list(resnet.children())[0:3]),
            nn.Sequential(*list(resnet.children())[3:5])
        )

        self.encoder2 = nn.Sequential(
            nn.Sequential(*list(resnet.children())[5:6]),
            nn.Sequential(*list(resnet.children())[6:7]),
            nn.Sequential(*list(resnet.children())[7:8])
        )

        self.aspp = ASPP(in_channels=2048)

        self.high_block = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.low_block = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.attention = AttentionFusion(in_channels=256)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        low_feature = self.encoder1(x)

        x = self.encoder2(low_feature)
        high_feature = self.aspp(x)

        high_feature = self.high_block(high_feature)
        low_feature = self.low_block(low_feature)

        middle_feature = self.attention(high_feature, low_feature)

        decoder1_feature = self.decoder1(middle_feature)
        decoder2_feature = self.decoder2(decoder1_feature)
        output_feature = self.decoder3(decoder2_feature)

        return output_feature


if __name__ == "__main__":
    model = AttentionDeeplabV3plus(num_classes=1)
    # model_base_path = r""
    # model.load_state_dict(torch.load(model_base_path))
    summary(model, input_size=(3, 256, 256), device="cpu")
