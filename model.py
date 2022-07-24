"""
@author: Jebearssica

"""
from torch import nn, cat, Tensor
from torch.nn import functional as F


class AdaptiveNorm(nn.Module):

    def __init__(self, n):

        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(Tensor([1.0]))
        self.w_1 = nn.Parameter(Tensor([0.0]))

        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.0001)

    def forward(self, x):

        return self.w_0 * x + self.w_1 * self.bn(x)

class LFN(nn.Module):
    """
    our LFN with 3 Linear fusion blocks
    """

    def __init__(self, c, norm=AdaptiveNorm):

        super(LFN, self).__init__()
        self.LNB1 = nn.Sequential(
            nn.Conv2d(6, c, kernel_size=3, stride=1,
                      padding=1, dilation=1, bias=False),
            norm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1,
                      padding=2, dilation=2, bias=False),
            norm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1,
                      padding=4, dilation=4, bias=False),
            norm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1,
                      padding=8, dilation=8, bias=False),
            norm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1,
                      padding=16, dilation=16, bias=False),
            norm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1,
                      padding=32, dilation=32, bias=False),
            norm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1,
                      padding=1, dilation=1, bias=False),
            norm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, 3, kernel_size=1, stride=1, padding=0, dilation=1),
        )

        self.LNB2 = nn.Sequential(
            nn.Conv2d(3, c, 3, bias=False,
                      padding=1, dilation=1),
            norm(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, bias=False,
                      padding=1, dilation=1),
            norm(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 3, 1),
        )

        self.LNB3 = nn.Sequential(
            nn.Conv2d(6, c, kernel_size=3, stride=1,
                      padding=1, dilation=1, bias=False),
            norm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1,
                      padding=2, dilation=2, bias=False),
            norm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1,
                      padding=4, dilation=4, bias=False),
            norm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1,
                      padding=8, dilation=8, bias=False),
            norm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1,
                      padding=16, dilation=16, bias=False),
            norm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1,
                      padding=32, dilation=32, bias=False),
            norm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1,
                      padding=1, dilation=1, bias=False),
            norm(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, 3, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.Sigmoid()
        )

    def forward(self, lrRF, hrFF):
        _, _, h_lrx, w_lrx = lrRF.size()
        _, _, h_hrx, w_hrx = hrFF.size()

        LrGuidedUp = F.interpolate(
            lrRF, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        HrDown = F.interpolate(
            hrFF, (h_lrx, w_lrx), mode='bilinear', align_corners=True)

        xLr = self.LNB1(cat([lrRF, HrDown], dim=1))
        A1 = F.interpolate(
            xLr, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        output1 = (1-A1)*LrGuidedUp+A1*hrFF

        A2 = self.LNB2(output1)

        output2 = (1-A2)*LrGuidedUp+A2*hrFF

        A3 = self.LNB3(cat([output1, output2], dim=1))

        output3 = (1-A3)*LrGuidedUp+A3*hrFF

        return output3

