import megengine.module as nn
import megengine.functional as F
import numpy as np
from collections import OrderedDict
from megengine.utils.module_stats import module_stats

relu = nn.GELU


def Conv2D(
        in_channels: int, out_channels: int,
        kernel_size: int, stride: int, padding: int,
        is_seperable: bool = False, has_relu: bool = False,
):
    modules = OrderedDict()

    if is_seperable:
        modules['depthwise'] = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False,
        )
        modules['pointwise'] = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=True,
        )
    else:
        modules['conv'] = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=True,
        )
    if has_relu:
        modules['relu'] = relu()
    return nn.Sequential(modules)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False, shortcut=False):
        super(CALayer, self).__init__()
        self.shortcut = shortcut
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        weight = self.conv_du(y)
        return weight


class SALayer(nn.Module):
    def __init__(self, kernel_size=3):
        super(SALayer, self).__init__()
        self.compress = lambda x: F.concat((F.max(x, 1, keepdims=True), F.mean(x, 1, keepdims=True)), axis=1)
        self.spatial = Conv2D(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, is_seperable=False,
                              has_relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = self.compress(x)
        x = self.spatial(x)
        x = F.nn.sigmoid(x)  # broadcasting
        return x


class SkipDAB(nn.Module):
    def __init__(self, n_feat, reduction, kernel_size=1, bn=False):
        super(SkipDAB, self).__init__()
        self.SA = SALayer()  ## Spatial Attention
        self.CA = CALayer(n_feat, reduction)  ## Channel Attention
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)

    def forward(self, skip, up):
        sa_branch = self.SA(skip * up) * skip
        ca_branch = self.CA(skip * up) * skip
        res = F.concat([sa_branch, ca_branch], axis=1)
        res = self.conv1x1(res)
        return res


class DAB(nn.Module):
    def __init__(self, n_feat, reduction, kernel_size=3):
        super(DAB, self).__init__()
        self.body = nn.Sequential(
            Conv2D(n_feat, n_feat, kernel_size, stride=1, padding=(kernel_size - 1) // 2, is_seperable=True,
                   has_relu=True),
            #             Conv2D(n_feat, n_feat, kernel_size, stride=1, padding=(kernel_size - 1) // 2, is_seperable=True, has_relu=False)
        )
        self.SA = SALayer(kernel_size)  ## Spatial Attention
        self.CA = CALayer(n_feat, reduction)  ## Channel Attention
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)

    def forward(self, inp):
        x = self.body(inp)
        sa_branch = self.SA(x) * x
        ca_branch = self.CA(x) * x
        res = F.concat([sa_branch, ca_branch], axis=1)
        res = self.conv1x1(res)
        return res + inp


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = Conv2D(in_channels, mid_channels, kernel_size=5, stride=stride, padding=2, is_seperable=True,
                            has_relu=True)
        self.conv2 = Conv2D(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, is_seperable=True,
                            has_relu=True)
        self.proj = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels else
            Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, is_seperable=True,
                   has_relu=False)
        )

    def forward(self, x):
        proj = self.proj(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + proj
        return x


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction=8, down_sample=False):
        super(EncoderStage, self).__init__()
        stride = 2 if down_sample else 1
        self.down = EncoderBlock(in_channels=in_channels, mid_channels=out_channels // 4, out_channels=out_channels,
                                 stride=stride)

        self.conv = nn.Sequential(
            EncoderBlock(
                in_channels=out_channels,
                mid_channels=out_channels // 4,
                out_channels=out_channels,
                stride=1,
            ),
        )

        self.dab = nn.Sequential(
            DAB(out_channels, reduction, 3),
        )

    def forward(self, inp):
        inp = self.down(inp)
        x = self.conv(inp)
        x = self.dab(x)
        return x + inp


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, has_relu=True):
        super().__init__()
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_seperable=True,
                            has_relu=True)
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, is_seperable=True,
                            has_relu=has_relu)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.conv2(x)
        x = x + inp
        return x


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_in_channels: int, out_channels: int, reduction=8, bias=False, attn=False,
                 shortcut=False, up_sample=False, skip=True):
        super().__init__()
        self.decode_conv = DecoderBlock(in_channels, in_channels, True)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2,
                                           padding=0) if up_sample else Conv2D(in_channels, out_channels, kernel_size=1,
                                                                               stride=1, padding=0, is_seperable=False,
                                                                               has_relu=False)
        self.proj_conv = Conv2D(skip_in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_seperable=True,
                                has_relu=False)
        self.attn = SkipDAB(out_channels, reduction, bias, shortcut) if attn else None
        self.skip = skip

    def forward(self, inp, skip):
        up = self.upsample(self.decode_conv(inp))
        if self.skip:
            skip = self.proj_conv(skip)
            up = self.attn(skip, up) + up if self.attn else skip + up
        return up


class DAB1skip1DownSampleGeLU(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Sequential(
            Conv2D(1, 16, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True),
            DAB(n_feat=16, reduction=2, kernel_size=3),
            DAB(n_feat=16, reduction=2, kernel_size=3)
        )

        self.enc1 = EncoderStage(in_channels=16, out_channels=32, reduction=2, down_sample=True)
        self.enc2 = EncoderStage(in_channels=32, out_channels=32, reduction=4, )
        self.enc3 = EncoderStage(in_channels=32, out_channels=64, reduction=4)
        self.enc4 = EncoderStage(in_channels=64, out_channels=64, reduction=8)

        self.dec1 = DecoderStage(in_channels=64, skip_in_channels=64, out_channels=64, reduction=8, bias=False,
                                 attn=True, shortcut=False, skip=False)
        self.dec2 = DecoderStage(in_channels=64, skip_in_channels=32, out_channels=32, reduction=4, bias=False,
                                 attn=True, shortcut=False, skip=False)
        self.dec3 = DecoderStage(in_channels=32, skip_in_channels=32, out_channels=32, reduction=4, bias=False,
                                 attn=True, shortcut=False, skip=False)
        self.dec4 = DecoderStage(in_channels=32, skip_in_channels=16, out_channels=16, reduction=2, bias=False,
                                 attn=False, shortcut=False, skip=True, up_sample=True)
        self.dba = nn.Sequential(
            Conv2D(16, 16, kernel_size=3, padding=1, stride=1, is_seperable=True, has_relu=True),
            DAB(n_feat=16, reduction=2, kernel_size=3),
            DAB(n_feat=16, reduction=2, kernel_size=3),
        )
        self.out = nn.Sequential(
            Conv2D(16, 1, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=False),
        )

    def forward(self, inp):
        conv0 = self.conv0(inp)
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)

        conv4 = self.enc4(conv3)

        up3 = self.dec1(conv4, conv3)
        up2 = self.dec2(up3, conv2)
        up1 = self.dec3(up2, conv1)
        up0 = self.dec4(up1, conv0)
        up0 = self.dba(up0)
        x = self.out(up0)
        pred = inp + x
        return pred


if __name__ == "__main__":
    model = DAB1skip1DownSampleGeLU()
    input_data = np.random.rand(1, 1, 256, 256).astype("float32")
    total_stats, stats_details = module_stats(
        model,
        inputs=(input_data),
        cal_params=True,
        cal_flops=True,
        logging_to_stdout=False,
    )

    print("params %.3fK MAC/pixel %.0f" % (
        total_stats.param_dims / 1e3, total_stats.flops / input_data.shape[2] / input_data.shape[3]))
