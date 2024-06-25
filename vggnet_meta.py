import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import mmd_1
import torch.nn.functional as F
import torch


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class Conv2dML(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2dML, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                       bias=bias)
        self.weight.fast = None
        if self.bias is not None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2dML, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2dML, self).forward(x)
        return out


class BatchNorm2dML(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm2dML, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var,
                               weight, bias, training=self.training, momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device),
                               torch.ones(x.size(1), dtype=x.dtype, device=x.device),
                               weight, bias, training=True, momentum=1)
        return out


class BatchNorm1dML(nn.BatchNorm1d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm1dML, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device),
                               torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True,
                               momentum=1)
        return out


class LinearML(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearML, self).__init__(in_features, out_features, bias=bias)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(LinearML, self).forward(x)
        return out


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            LinearML(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            LinearML(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            LinearML(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2dML):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2dML):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Conv2dML):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2dML(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, BatchNorm2dML(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(num_classes, pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), num_classes, **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
        model.load_state_dict(torch.load("./model/vgg11_bn-6002323d.pth"), strict=False)
    return model


class my_vgg(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super(my_vgg, self).__init__()
        self.feature_layers = vgg(1000, pretrained=pretrained)
        self.cls_fc = nn.Linear(1000, num_classes)

    def forward(self, source):
        x = self.feature_layers(source)
        x = self.cls_fc(x)
        return x


class DSAN(nn.Module):

    def __init__(self, num_classes=5):
        super(DSAN, self).__init__()
        self.feature_layers = vgg(1000, pretrained=False)
        self.cls_fc = LinearML(1000, num_classes)


    # mode这个参数决定了正负情感分别有哪些，不同的库之间的正负情感不一样
    def forward(self, source, target, s_label, mode):
        # loss_mmd = 0
        source = self.feature_layers(source)
        s_pred = self.cls_fc(source)
        if self.training ==True:
            target = self.feature_layers(target)
            t_label = self.cls_fc(target)
            loss_mmd = mmd_1.mmd_rbf_noaccelerate(source, target)
            loss_lmmd = mmd_1.lmmd(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1))
            loss_nmmd = mmd_1.nmmd(source, target, s_label, torch.nn.functional.softmax(t_label, dim=1), mode=mode)

        else:
            loss_mmd, loss_lmmd, loss_nmmd = 0, 0, 0
        if self.training ==True:
            return s_pred, t_label, loss_mmd, loss_lmmd, loss_nmmd
        else: # 测试是不返回t_label
            return s_pred, loss_mmd, loss_lmmd, loss_nmmd
        # return s_pred, t_label  # 调试时修改此处了return s_pred, loss


if __name__ == '__main__':
    import torch
    x = torch.randn(64, 3, 128, 256)
    model = my_vgg(5, pretrained=True)
    print(model)
    x = model(x)
    print(x)
