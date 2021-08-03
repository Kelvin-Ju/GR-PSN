import sys
sys.path.append("./")
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from models import model_utils
import numpy as np
import pandas as pd
from collections import OrderedDict

class Encoder(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(Encoder, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64, k = 3, stride = 1, pad = 1)
        self.conv2 = model_utils.conv(batchNorm, 64, 128, k = 3, stride = 2, pad = 1)
        self.conv3 = model_utils.conv(batchNorm, 128, 128, k = 3, stride = 1, pad = 1)
        self.conv4 = model_utils.conv(batchNorm, 128, 256, k = 3, stride = 2, pad = 1)
        self.conv5 = model_utils.conv(batchNorm, 256, 256, k = 3, stride = 1, pad = 1)
    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out_feat = self.conv5(out)
        n, c, h, w = out_feat.data.shape
        return out_feat, [n, c, h, w]

class Decoder(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(Decoder, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64, k = 3, stride = 1, pad = 1)
        self.conv2 = model_utils.deconv(64, 128)
        self.conv3 = model_utils.conv(batchNorm, 128, 128, k = 3, stride = 1, pad = 1)
        self.conv4 = model_utils.deconv(128, 256)
        self.conv5 = model_utils.conv(batchNorm, 256, 3, k = 3, stride = 1, pad = 1)
    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out_feat = self.conv5(out)
        return out_feat

class RenderingNet(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=99, other={}):
        super(RenderingNet, self).__init__()
        self.encoder = Encoder(batchNorm, 6 , other)
        self.decoder = Decoder(batchNorm, 356,other)
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if len(x) > 1: # Have lighting
            light = x[1]
            normal=x[0]
            mtrl=x[2]
            light_split = torch.split(light, 3, 1)
            mtrl=mtrl.view(-1,100,1,1)

        outs=[]
        for i in range(len(light_split)):
            net_in = torch.cat([normal, light_split[i]], 1)
            feat, shape = self.encoder(net_in)
            # print(feat.shape[2:])
            mtrl = mtrl.expand(-1, 100,feat.shape[2],feat.shape[3])
            # print(feat.shape,mtrl.shape)
            out = self.decoder(torch.cat((feat,mtrl),1))
            outs.append(out)
#         print(out.shape)
        reimg = torch.cat(outs, 1)
        return reimg


class FeatExtractor(nn.Module) :
    def __init__(self, batchNorm=False, c_in=3, other={}) :
        super(FeatExtractor, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64, k = 3, stride = 1, pad = 1)
        self.conv2 = model_utils.conv(batchNorm, 64, 64, k = 3, stride = 1, pad = 1)
        self.conv3 = model_utils.conv(batchNorm, 64, 64, k = 3, stride = 1, pad = 1)
        self.conv4 = model_utils.conv(batchNorm, 64, 64, k = 3, stride = 1, pad = 1)
        self.conv5 = model_utils.conv(batchNorm, 64, 64, k = 3, stride = 1, pad = 1)
        self.conv6 = model_utils.conv(batchNorm, 64, 64, k = 3, stride = 1, pad = 1)

        self.conv7 = model_utils.conv(batchNorm, 64, 128, k = 3, stride = 2, pad = 1)  # from c2
        self.conv8 = model_utils.conv(batchNorm, 128, 128, k = 3, stride = 1, pad = 1)
        self.conv9 = model_utils.conv(batchNorm, 128, 128, k = 3, stride = 1, pad = 1)
        self.conv10 = model_utils.conv(batchNorm, 128, 128, k = 3, stride = 1, pad = 1)

        self.conv11 = model_utils.conv(batchNorm, 128, 256, k = 3, stride = 2, pad = 1)  # from c8
        self.conv12 = model_utils.conv(batchNorm, 256, 256, k = 3, stride = 1, pad = 1)
        self.conv13 = model_utils.conv(batchNorm, 256, 256, k = 3, stride = 1, pad = 1)

        self.conv3down1 = model_utils.conv(batchNorm, 64, 128, k = 3, stride = 2, pad = 1)  # to c8
        self.conv4down1 = model_utils.conv(batchNorm, 64, 128, k = 3, stride = 2, pad = 1)  # to c9
        self.conv4down2 = model_utils.conv(batchNorm, 64, 256, k = 3, stride = 4, pad = 1)  # to c11
        self.conv5down1 = model_utils.conv(batchNorm, 64, 128, k = 3, stride = 2, pad = 1)  # to c10
        self.conv5down2 = model_utils.conv(batchNorm, 64, 256, k = 3, stride = 4, pad = 1)  # to c12
        self.conv10down1 = model_utils.conv(batchNorm, 128, 256, k = 3, stride = 2, pad = 1)  # to c13
        self.convnor8 = model_utils.conv(batchNorm, 128, 64, k = 1, stride = 1, pad = 0)
        self.convnor122 = model_utils.conv(batchNorm, 256, 128, k = 1, stride = 1, pad = 0)
        self.convnor124 = model_utils.conv(batchNorm, 256, 64, k = 1, stride = 1, pad = 0)
        self.convnor10 = model_utils.conv(batchNorm, 128, 64, k = 1, stride = 1, pad = 0)

    def forward(self, x) :
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out3d = self.conv3down1(out3)

        out7 = self.conv7(out2)
        out8 = self.conv8(out7)

        out8up = torch.nn.functional.upsample(out8, scale_factor = 2, mode = 'bilinear', align_corners = True)
        out8upnor = self.convnor8(out8up)

        out8 = torch.add(out8, out3d)

        out4 = self.conv4(out3)
        out4 = torch.add(out4, out8upnor)
        out4d1 = self.conv4down1(out4)
        out4d2 = self.conv4down2(out4)

        out9 = self.conv9(out8)
        out9 = torch.add(out9, out4d1)

        out5 = self.conv5(out4)
        out5d1 = self.conv5down1(out5)
        out5d2 = self.conv5down2(out5)
        out11 = self.conv11(out8)
        out11 = torch.add(out11, out4d2)

        out12 = self.conv12(out11)
        out12 = torch.add(out12, out5d2)
        out12up2 = torch.nn.functional.upsample(out12, scale_factor = 2, mode = 'bilinear', align_corners = True)
        out12up2nor = self.convnor122(out12up2)
        out12up4 = torch.nn.functional.upsample(out12, scale_factor = 4, mode = 'bilinear', align_corners = True)
        out12up4nor = self.convnor124(out12up4)

        out10 = self.conv10(out9)
        out10 = torch.add(out10, out5d1)
        out10 = torch.add(out10, out12up2nor)
        out10d = self.conv10down1(out10)
        out10up = torch.nn.functional.upsample(out10, scale_factor = 2, mode = 'bilinear', align_corners = True)
        out10upnor = self.convnor10(out10up)

        out13 = self.conv13(out12)
        out13 = torch.add(out13, out10d)

        out6 = self.conv6(out5)
        out6 = torch.add(out6, out12up4nor)
        out6 = torch.add(out6, out10upnor)

        out6m = out6
        out10m = out10
        out13m = out13
        n6, c6, h6, w6 = out6m.data.shape
        out6m = out6m.view(-1)

        n10, c10, h10, w10 = out10m.data.shape
        out10m = out10m.view(-1)

        n13, c13, h13, w13 = out13m.data.shape
        out13m = out13m.view(-1)

        return out6m, [n6, c6, h6, w6], out10m, [n10, c10, h10, w10], out13m, [n13, c13, h13, w13]


class _DenseLayer(nn.Sequential) :
    def __init__(self, in_channels, growth_rate, bn_size) :
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace = True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate,
                                           kernel_size = 1,
                                           stride = 1, bias = False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace = True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size = 3,
                                           stride = 1, padding = 1, bias = False))

    # ÷ÿ‘ÿforward∫Ø ˝
    def forward(self, x) :
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential) :
    def __init__(self, num_layers, in_channels, bn_size, growth_rate) :
        super(_DenseBlock, self).__init__()
        for i in range(num_layers) :
            self.add_module('denselayer%d' % (i + 1),
                            _DenseLayer(in_channels + growth_rate * i,
                                        growth_rate, bn_size))


class _Transition(nn.Sequential) :
    def __init__(self, in_channels, out_channels) :
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace = True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size = 1,
                                          stride = 1, bias = False))
        self.add_module('pool', nn.AvgPool2d(kernel_size = 1, stride = 1))


class DenseNet_BC(nn.Module) :
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16),
                 bn_size=4, theta=0.5, num_classes=10) :
        super(DenseNet_BC, self).__init__()

        # ≥ı ºµƒæÌª˝Œ™filter:2±∂µƒgrowth_rate
        num_init_feature = 2 * growth_rate

        # ±Ì æcifar-10
        if num_classes == 10 :
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_feature,
                                    kernel_size = 7, stride = 1,
                                    padding = 3, bias = False)),
            ]))
        else :
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_feature,
                                    kernel_size = 3, stride = 1,
                                    padding = 1, bias = False)),
                ('norm0', nn.BatchNorm2d(num_init_feature)),
                ('relu0', nn.ReLU(inplace = True)),
                ('pool0', nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
            ]))

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config) :
            self.features.add_module('denseblock%d' % (i + 1),
                                     _DenseBlock(num_layers, num_feature,
                                                 bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config) - 1 :
                self.features.add_module('transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.features.add_module('relu5', nn.ReLU(inplace = True))
        # self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))

        # self.classifier = nn.Linear(num_feature, num_classes)

        for m in self.modules() :
            if isinstance(m, nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) :
                nn.init.constant_(m.bias, 0)

    def forward(self, x) :
        # features = self.features(x)
        # out = features.view(features.size(0), -1)
        # out = self.classifier(out)
        out = self.features(x)
        return out


# DenseNet_BC for ImageNet
def DenseNet121() :
    # return DenseNet_BC(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=1000)
    return DenseNet_BC(growth_rate = 32, block_config = (1, 2, 4, 3), num_classes = 1000)


def DenseNet169() :
    return DenseNet_BC(growth_rate = 32, block_config = (6, 12, 32, 32), num_classes = 1000)


def DenseNet201() :
    return DenseNet_BC(growth_rate = 32, block_config = (6, 12, 48, 32), num_classes = 1000)


def DenseNet161() :
    return DenseNet_BC(growth_rate = 48, block_config = (6, 12, 36, 24), num_classes = 1000, )


# DenseNet_BC for cifar
def densenet_BC_100() :
    return DenseNet_BC(growth_rate = 12, block_config = (16, 16, 16))


class FeatExtractor1(nn.Module) :
    def __init__(self, batchNorm=False, c_in=3, other={}) :
        super(FeatExtractor1, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64, k = 3, stride = 1, pad = 1)
        self.conv2 = model_utils.conv(batchNorm, 64, 128, k = 3, stride = 2, pad = 1)
        self.conv3 = model_utils.conv(batchNorm, 128, 128, k = 3, stride = 1, pad = 1)

    def forward(self, x) :
        # print('x:',x.shape)
        # out = self.conv1(x)
        #  print('1:',out.shape)
        # out = self.conv2(out)
        # print('2:',out.shape)
        # out_feat = self.conv3(out)
        # print('3:',out_feat.shape)
        # n, c, h, w = out_feat.data.shape
        #  out_feat   = out_feat.view(-1)
        out = self.conv1(x)
        # print('1:',out.shape)
        out = self.conv2(out)
        # print('2:',out.shape)
        out_feat = self.conv3(out)
        #  print('3:',out_feat.shape)
        n, c, h, w = out_feat.data.shape
        out_feat = out_feat.view(-1)
        return out_feat, [n, c, h, w]


class FeatExtractor2(nn.Module) :
    def __init__(self, batchNorm=False, other={}) :
        super(FeatExtractor2, self).__init__()
        self.other = other
        self.conv4 = model_utils.conv(batchNorm, 256, 256, k = 3, stride = 1, pad = 1)
        self.conv5 = model_utils.conv(batchNorm, 256, 256, k = 3, stride = 1, pad = 1)
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k = 3, stride = 1, pad = 1)

    def forward(self, x) :
        out = self.conv4(x)
        out = self.conv5(out)
        out = self.conv6(out)
        out_feat = self.conv7(out)
        n, c, h, w = out_feat.data.shape
        out_feat = out_feat.view(-1)
        return out_feat, [n, c, h, w]


class Regressor(nn.Module) :
    def __init__(self, batchNorm=False, other={}) :
        super(Regressor, self).__init__()
        self.other = other
        self.deconv1 = model_utils.conv(batchNorm, 128, 128, k = 3, stride = 1, pad = 1)
        self.deconv2 = model_utils.deconv(128, 64)
        self.deconv3 = self._make_output(64, 3, k = 3, stride = 1, pad = 1)
        self.deconv4 = DenseNet121()
        self.deconv5 = model_utils.deconv(188, 64)
        self.deconv6 = model_utils.conv(batchNorm, 64, 64, k = 3, stride = 2, pad = 1)
        self.est_normal = self._make_output(64, 3, k = 3, stride = 1, pad = 1)
        self.other = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1) :
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size = k, stride = stride, padding = pad, bias = False))

    def forward(self, x, shape) :
        x = x.view(shape[0], shape[1], shape[2], shape[3])
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.deconv6(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal


class GeometryNet(nn.Module) :
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}) :
        super(GeometryNet, self).__init__()
        self.extractor1 = FeatExtractor1(batchNorm, c_in, other)
        self.extractor2 = FeatExtractor2(batchNorm, other)
        self.regressor = Regressor(batchNorm, other)
        self.c_in = c_in
        self.fuse_type = fuse_type
        self.other = other

        for m in self.modules() :
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) :
                kaiming_normal_(m.weight.data)
                if m.bias is not None :
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) :
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x) :
        img = x[0]
        img_split = torch.split(img, 3, 1)
        if len(x) > 1 :  # Have lighting
            light = x[1]
            light_split = torch.split(light, 3, 1)

        feats = []
        for i in range(len(img_split)) :
            net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
            # m =nn.Upsample(scale_factor=4, mode='nearest')
            # m(net_in)
            feat, shape = self.extractor1(net_in)
            feats.append(feat)
        if self.fuse_type == 'mean' :
            feat_fused = torch.stack(feats, 1).mean(1)
        elif self.fuse_type == 'max' :
            feat_fused, _ = torch.stack(feats, 1).max(1)

        featss = []
        for i in range(len(img_split)) :
            net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
            # m(net_in)
            feat, shape = self.extractor1(net_in)
            feat = feat.view(shape[0], shape[1], shape[2], shape[3])
            feat_fused = feat_fused.view(shape[0], shape[1], shape[2], shape[3])
            featt = torch.cat((feat, feat_fused), 1)
            featt, shapee = self.extractor2(featt)
            featss.append(featt)
        if self.fuse_type == 'mean' :
            feat_fusedd = torch.stack(featss, 1).mean(1)
        elif self.fuse_type == 'max' :
            feat_fusedd, _ = torch.stack(featss, 1).max(1)
        normal = self.regressor(feat_fusedd, shapee)
        return normal


if __name__ =="__main__":
    # 测试 RenderingNet
    testTensor = torch.randn(1, 3, 32, 32)
    mtrl = torch.randn(1, 100)
    light = torch.randn(1, 96, 32, 32)

    x = [testTensor, light, mtrl]
    net = RenderingNet('max', False, 3, {'img_num' : 32, 'in_light' : True})
    #
    ouput = net(x)
    #
    print(ouput.shape)

    # 测试 GeometryNet
    testTensor = torch.randn(1, 96, 32, 32)
    mtrl = torch.randn(1, 100)
    light = torch.randn(1, 96, 32, 32)

    x = []
    x.append(testTensor)
    x.append(light)
    net = GeometryNet('max', False, 6, {'img_num' : 32, 'in_light' : True})
    #
    ouput = net(x)
    #
    print(ouput.shape)