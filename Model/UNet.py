import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class conv3x3_block_x1(nn.Module):
    '''(conv => BN => ReLU) * 1'''

    def __init__(self, in_ch, out_ch):
        super(conv3x3_block_x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv3x3_block_x2(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(conv3x3_block_x2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upsample, self).__init__()
        self.conv1x1 = conv1x1(in_ch, out_ch)
        self.conv = conv3x3_block_x2(in_ch, out_ch)

    def forward(self, H, L):
        """
        H: High level feature map, upsample
        L: Low level feature map, block output
        """
        H = F.interpolate(H, scale_factor=2, mode='bilinear', align_corners=False)
        H = self.conv1x1(H)
        x = torch.cat([H, L], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.block1 = conv3x3_block_x2(3, 64)
        self.block2 = conv3x3_block_x2(64, 128)
        self.block3 = conv3x3_block_x2(128, 256)
        self.block4 = conv3x3_block_x2(256, 512)
        self.block_out = conv3x3_block_x1(512, 1024)
        self.upsample1 = upsample(1024, 512)
        self.upsample2 = upsample(512, 256)
        self.upsample3 = upsample(256, 128)
        self.upsample4 = upsample(128, 64)
        self.upsample_out = conv3x3_block_x2(64, num_classes)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        block1_x = self.block1(x)
        x = self.maxpool(block1_x)
        block2_x = self.block2(x)
        x = self.maxpool(block2_x)
        block3_x = self.block3(x)
        x = self.maxpool(block3_x)
        block4_x = self.block4(x)
        x = self.maxpool(block4_x)
        x = self.block_out(x)
        x = self.upsample1(x, block4_x)
        x = self.upsample2(x, block3_x)
        x = self.upsample3(x, block2_x)
        x = self.upsample4(x, block1_x)
        x = self.upsample_out(x)
        x = self.Sigmoid(x)

        return x


if __name__ == "__main__":
    model = UNet(num_classes=1)
    # model_base_path = r""
    # model.load_state_dict(torch.load(model_base_path))
    summary(model, input_size=(3, 256, 256), device="cpu")
