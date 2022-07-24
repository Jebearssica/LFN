import torch.cuda
import torch.nn.functional as F
from time import perf_counter


def forward(input, model):
    lrRF, hrFF, GT = input[:3]
    with torch.cuda.device(0):
        lrRF, hrFF, GT = lrRF.cuda(), hrFF.cuda(), GT.cuda()
    yHat = model(lrRF, hrFF)
    return yHat, GT

def evalForward(imgs, model):
    length = len(imgs)
    with torch.cuda.device(0):
        imgs = [imgs[index].cuda() for index in range(0, length)]
        model.cuda()
    timeStart = perf_counter()
    y_hat = model(imgs[0], imgs[1])
    deltaTime = perf_counter()-timeStart
    for index in range(2, length-1):
        timeStart = perf_counter()
        y_hat = model(y_hat, imgs[index])
        deltaTime += perf_counter()-timeStart

    return y_hat, imgs[length-1], deltaTime