from utils.forward import forward
from model import LFN
import os.path
import torch.cuda
from torch import load, nn, save
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from dataset import pngDataset, Config


def train(config):
    # load model
    if os.path.exists(config.MODELPATH):
        config.MODEL.load_state_dict(load(config.MODELPATH))

    dataset = pngDataset(config.DATASET, csvType=config.TYPE)
    testLoader = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                            num_workers=config.NUMBER_WORKER, shuffle=True)

    # loss function
    criterion = nn.MSELoss()

    # optimizer
    optimizer = optim.Adam(config.MODEL.parameters(), lr=0.001)

    writer = SummaryWriter()

    with torch.cuda.device(0):
        config.MODEL.cuda()
        criterion.cuda()

    loss = 0

    for epoch in range(config.N_START, config.N_EPOCH):
        totalLoss = 0

        for _, imgs in enumerate(testLoader):

            y, gt = config.FORWARD(imgs[0], config.MODEL)

            loss = criterion(y, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            totalLoss += loss

        save(config.MODEL.state_dict(),
             config.MODELSAVE+'{}.pth'.format(epoch))

        writer.add_scalar('singleLoss', loss, epoch)
        writer.add_scalar('totalLoss', totalLoss/len(testLoader), epoch)
        if epoch % config.STEP == 0:
            writer.add_images('output', y, epoch)
            writer.add_images('GT', gt, epoch)


if __name__ == "__main__":
    trainConfig = Config(
        DATASET='./Data/TrainDataset/', # dataset dir
        MODELPATH='./Model/',           # pretrained model dir (if exist)
        MODELSAVE='./Model/',           # model save dir
        BATCH_SIZE=16,
        NUMBER_WORKER=0,
        STEP=25,                        # every 25 steps output image to tensorboard
        N_START=1,                      # train epoch: [N_START, N_EPOCH]
        N_EPOCH=10001,
        TYPE='train',                   # dataset .csv file name
        FORWARD=forward,
        MODEL=LFN(c=32),
    )
    train(trainConfig)
