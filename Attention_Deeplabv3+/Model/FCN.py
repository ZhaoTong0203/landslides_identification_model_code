import timm
import torch
import torch.nn as nn
from torchsummary import summary


class FCN8(nn.Module):
    def __init__(self, num_classes):
        super(FCN8, self).__init__()

        encoder = timm.create_model("tv_resnet50", pretrained=True,
                                    features_only=True, in_chans=3, output_stride=8)

        # Encoder
        self.encoder = nn.Sequential(*list(encoder.children())[0:3])
        self.fc1 = nn.Sequential(*list(encoder.children())[3:5])
        self.fc2 = nn.Sequential(*list(encoder.children())[5:7])
        self.fc3 = nn.Sequential(*list(encoder.children())[7:8])

        # Decoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv_trans1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_trans2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.conv_trans3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_output = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)
        x2 = self.fc1(x1)
        x3 = self.fc2(x2)
        x4 = self.fc3(x3)

        # Decoder
        x5 = self.conv1(x4)
        x6 = self.conv2(x5)
        x7 = self.conv3(x6)

        # Skip connections
        x8 = self.conv_trans1(x7)
        x8 = x2 + x8 

        x9 = self.conv_trans2(x8)
        x9 = x1 + x9

        x10 = self.conv_trans3(x9)
        x10 = self.conv_output(x10)

        output = self.sigmoid(x10)

        return output


if __name__ == "__main__":
    model = FCN8(num_classes=1)
    model_base_path = r"../Model_save/fcn8_model_params.pth"
    model.load_state_dict(torch.load(model_base_path))
    summary(model, input_size=(3, 256, 256), device="cpu")
