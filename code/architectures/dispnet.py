import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_planes, track_running_stats=False),
        # nn.Dropout(0.2),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
        
    )


def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )

def predict_disp_final(in_planes, out_channels = 3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_channels, kernel_size=3, padding=1),
        # nn.Sigmoid()
    )

def predict_disp_unc(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 3, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        # nn.Sigmoid()
    )

def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )

 
def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


class DispNetS(nn.Module):

    def __init__(self, out_channels = 3, sigmoid = True, alpha=50, beta=0.01):
        super(DispNetS, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.sigmoid = sigmoid

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv(3,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp_final(upconv_planes[6], out_channels)
        self.predict_disp_unc = predict_disp_unc(upconv_planes[6])
        self.sigmoid = nn.Sigmoid()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        # print("OI")
        out_conv1 = self.conv1(x)
        # out_conv1=F.dropout(out_conv1, p =0.2, training = True)
        out_conv2 = self.conv2(out_conv1)
        # out_conv2=F.dropout(out_conv2, p =0.2, training = True)
        out_conv3 = self.conv3(out_conv2)
        # out_conv3=F.dropout(out_conv3, p =0.2, training = True)
        out_conv4 = self.conv4(out_conv3)
        # out_conv4=F.dropout(out_conv4, p =0.2, training = True)
        out_conv5 = self.conv5(out_conv4)
        # out_conv5=F.dropout(out_conv5, p =0.2, training = True)
        out_conv6 = self.conv6(out_conv5)
        # out_conv6=F.dropout(out_conv6, p =0.2, training = True)
        out_conv7 = self.conv7(out_conv6)
        # out_conv7=F.dropout(out_conv7, p =0.2, training = True)

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)
        # out_iconv7=F.dropout(out_iconv7, p =0.2, training = True)

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)
        # out_iconv6=F.dropout(out_iconv6, p =0.2, training = True)

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)
        # out_iconv5=F.dropout(out_iconv5, p =0.2, training = True)

        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        # out_iconv4=F.dropout(out_iconv4, p =0.2, training = True)

        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        # out_iconv3=F.dropout(out_iconv3, p =0.2, training = True)

        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        # out_iconv2=F.dropout(out_iconv2,p =0.2)

        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        
        out_iconv1 = self.iconv1(concat1)
        # out_iconv1=F.dropout(out_iconv1, p =0.2, training = True)
        
        if self.sigmoid:
            out = self.sigmoid(self.predict_disp1(out_iconv1))
        else:
            out = self.predict_disp1(out_iconv1)
        
        return out