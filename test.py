from utils.forward import evalForward
from model import LFN
from utils.utils import psnr
import torch.cuda
from torch import load
from numpy import asarray_chkfinite
from torch.utils.data import DataLoader
import torchvision.transforms
from dataset import Config, pngDataset
import os.path
from skimage.metrics import structural_similarity
from cv2 import cvtColor, COLOR_BGR2GRAY
import csv

def evalPng(config):
    csvFile = open(file=config.SAVE+'{}.csv'.format(config.TYPE),
                   mode='w+', encoding='utf-8', newline="")
    csvWriter = csv.writer(csvFile)
    if os.path.exists(config.MODEL_LOAD_PATH):
        config.MODEL.load_state_dict(load(config.MODEL_LOAD_PATH))

    pngData = pngDataset(config.DATASET, config.TYPE, True)
    testLoader = DataLoader(pngData, batch_size=config.BATCH_SIZE,
                            num_workers=config.NUMBER_WORKER, shuffle=False)
    with torch.cuda.device(0):
        config.MODEL.cuda()
    index = 0
    for index, imgs in enumerate(testLoader):
        with torch.no_grad():
            config.MODEL.eval()

            y, gt, deltaTime = config.FORWARD(imgs[0], config.MODEL)
            y, gt = y.cpu(), gt.cpu()
            PILImage = torchvision.transforms.ToPILImage()(y.squeeze())
            PILImage.save(
                '{}{}-{}.png'.format(config.SAVE, config.TYPE, index))
            y_hat = asarray_chkfinite(PILImage)
            gtImg = asarray_chkfinite(
                torchvision.transforms.ToPILImage()(gt.squeeze()))
            psnrValue, _ = psnr(y_hat, gtImg)
            SSIM = structural_similarity(cvtColor(
                src=y_hat, code=COLOR_BGR2GRAY), cvtColor(src=gtImg, code=COLOR_BGR2GRAY))
            # write to csv
            csvWriter.writerow([psnrValue, SSIM, deltaTime])

def testAllInOne(config):
    """
    test x2, x4, x8, in deep & shallow DOF dataset
    """
    # select best model
    saveRoot = config.SAVE

    # eval shallow
    config.DATASET = './Data/TestDataset/ShallowDOF/'
    config.FORWARD = evalForward
    config.SAVE = saveRoot+'Shallow/'
    if not os.path.exists(config.SAVE):
        os.mkdir(config.SAVE)
    config.TYPE = 'x2'
    evalPng(config)
    config.FORWARD = evalForward
    config.TYPE = 'Pipex4'
    evalPng(config)
    config.TYPE = 'Pipex8'
    evalPng(config)

    # eval deep
    config.DATASET = './Data/TestDataset/DeepDOF/'
    config.FORWARD = evalForward
    config.SAVE = saveRoot+'Deep/'
    if not os.path.exists(config.SAVE):
        os.mkdir(config.SAVE)
    config.TYPE = 'x2'
    evalPng(config)
    config.FORWARD = evalForward
    config.TYPE = 'Pipex4'
    evalPng(config)
    config.TYPE = 'Pipex8'
    evalPng(config)

if __name__ == "__main__":
    evalPngConfig = Config(
        DATASET='',                         # test data dir
        MODEL_LOAD_PATH='./Model/LFN.pth',  # model dir
        SAVE='',                            # save output dir
        NUMBER_WORKER=1,
        BATCH_SIZE=1,
        TYPE='x8',                          # scale
        MODEL=LFN(c=32),
        FORWARD=evalForward,
    )
    # test all scale in one, including x2 x4 x8 in both deep & shallow DoF
    testAllInOne(evalPngConfig)
    # test in single step for x8
    evalPng(evalPngConfig)
