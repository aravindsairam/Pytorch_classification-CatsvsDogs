import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class fire(nn.Module):
    def __init__(self, in_chn, squeeze_chn, expand_chn, padding):
        super(fire, self).__init__()
        self.conv_s1 = nn.Conv2d(in_channels= in_chn,
                              out_channels= squeeze_chn,
                              kernel_size= 1)
        self.bn_s = nn.BatchNorm2d(squeeze_chn)

        self.conv_e1 = nn.Conv2d(in_channels= squeeze_chn,
                              out_channels= expand_chn,
                              kernel_size= 1)

        self.conv_e3 = nn.Conv2d(in_channels= squeeze_chn,
                              out_channels= expand_chn,
                              kernel_size= 3,
                              padding = padding)
        self.bn_e = nn.BatchNorm2d(expand_chn*2)

    def forward(self, x):
        s1 = F.relu(self.bn_s(self.conv_s1(x)))
        e1 = self.conv_e1(s1)
        e3 = self.conv_e3(s1)
        se_cat = F.relu(self.bn_e(torch.cat([e1, e3], dim = -3)))
        return se_cat


class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 3,
                              out_channels= 96,
                              kernel_size= 7,
                              stride = 2,
                              padding = 2)

        self.bn1 = nn.BatchNorm2d(96)

        self.maxpool1 = nn.MaxPool2d(kernel_size = 3,
                                    stride = 2)

        self.fire2 = fire(in_chn = 96, squeeze_chn = 16, expand_chn = 64, padding = 1)
        self.fire3 = fire(in_chn = 128, squeeze_chn = 16, expand_chn = 64, padding = 1)
        self.fire4 = fire(in_chn = 128, squeeze_chn = 32, expand_chn = 128, padding = 1)

        self.maxpool2 = nn.MaxPool2d(kernel_size = 3,
                                    stride = 2)

        self.fire5 = fire(in_chn = 256, squeeze_chn = 32, expand_chn = 128, padding = 1)
        self.fire6 = fire(in_chn = 256, squeeze_chn = 48, expand_chn = 192, padding = 1)
        self.fire7 = fire(in_chn = 384, squeeze_chn = 48, expand_chn = 192, padding = 1)
        self.fire8 = fire(in_chn = 384, squeeze_chn = 64, expand_chn = 256, padding = 1)

        self.maxpool3 = nn.MaxPool2d(kernel_size = 3,
                                    stride = 2)

        self.fire9 = fire(in_chn = 512, squeeze_chn = 64, expand_chn = 256, padding = 1)

        self.conv10 = nn.Conv2d(in_channels= 512,
                              out_channels= 1,
                              kernel_size= 1)

        self.bn2 = nn.BatchNorm2d(1)

        self.avgpool10 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        m1 = self.maxpool1(c1)
        f2 = self.fire2(m1)
        f3 = self.fire3(f2)
        f2_f3 = torch.add(f2, f3)
        f4 = self.fire4(f2_f3)
        m2 = self.maxpool2(f4)
        f5 = self.fire5(m2)
        m2_f5 = torch.add(m2, f5)
        f6 = self.fire6(m2_f5)
        f7 = self.fire7(f6)
        f6_f7 = torch.add(f6, f7)
        f8 = self.fire8(f6_f7)
        m3 = self.maxpool3(f8)
        f9 = self.fire9(m3)
        m3_f9 = F.dropout2d(torch.add(m3, f9), p =0.5)
        c10 = F.relu(self.bn2(self.conv10(m3_f9)))
        avg10 = self.avgpool10(c10)
        return torch.flatten(avg10)

class ResNet34(nn.Module):
    def __init__(self, pretrain = True):
        super(ResNet34, self).__init__()

        self.resnet = models.resnet34(pretrained=pretrain)

        self.resnet.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet(x)
        return torch.sigmoid(torch.flatten(x))

class ResNet50(nn.Module):
    def __init__(self, pretrain = True):
        super(ResNet50, self).__init__()

        self.resnet = models.resnet50(pretrained=pretrain)

        self.resnet.fc = nn.Linear(2048, 1)


    def forward(self, x):
        x = self.resnet(x)
        return torch.sigmoid(torch.flatten(x))

