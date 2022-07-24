import os.path
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms.transforms import ColorJitter
from torchvision.transforms.functional import adjust_brightness, adjust_hue, adjust_contrast, adjust_saturation, hflip, vflip
from torch import tensor, randperm, rand
from torch.nn import Module


"""
custom dataset reinforcement(override the official implementation)
"""


class ColorJitterList(ColorJitter):
    """
    random color jitter for lr+hr+gt imgs(override forward part of the official one)
    """

    def forward(self, imgs):
        """
        Args:
            img (PIL Image or Tensor) list: Input images.

        Returns:
            PIL Image or Tensor: Color jittered images.
        """
        fn_idx = randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = tensor(1.0).uniform_(
                    brightness[0], brightness[1]).item()
                imgs = [adjust_brightness(img, brightness_factor)
                        for img in imgs]

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = tensor(1.0).uniform_(
                    contrast[0], contrast[1]).item()
                imgs = [adjust_contrast(img, contrast_factor) for img in imgs]

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = tensor(1.0).uniform_(
                    saturation[0], saturation[1]).item()
                imgs = [adjust_saturation(img, saturation_factor)
                        for img in imgs]

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = tensor(1.0).uniform_(hue[0], hue[1]).item()
                imgs = [adjust_hue(img, hue_factor) for img in imgs]

        return imgs


class RandomFlipList(Module):
    """Vertically/Horizontal flip the given PIL Image randomly with a given probability.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, imgs):
        """
        Args:
            imgs (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Images or Tensors: Randomly flipped images.
        """
        if rand(1) < self.p:
            imgs = [vflip(img) for img in imgs]
        if rand(1) < self.p:
            imgs = [hflip(img) for img in imgs]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def defaultLoader(path):
    return Image.open(path).convert('RGB')


class Config(object):
    def __init__(self, **params):
        for k, v in params.items():
            self.__dict__[k] = v


class pngDataset(data.Dataset):

    def __init__(self, rootPath, csvType, isTest=False, loader=defaultLoader):
        super(pngDataset, self).__init__()

        # concatenate full path
        with open(rootPath+csvType+'.csv') as f:
            imgs = [line.strip().split(',') for line in f]
        imgs = [[os.path.join(rootPath, p) for p in group] for group in imgs]

        self.loader = loader
        self.imgs = imgs
        self.imgNames = [os.path.basename(imgs[index][0])
                         for index in range(0, len(imgs))]

        if isTest:
            self.transform = None
        else:
            self.transform = transforms.Compose([
                transforms.RandomApply(
                    [
                        ColorJitterList(1, 1, 1, 0.25)
                    ], p=0.5
                ),
                RandomFlipList(p=0.5)
            ])

    def get_path(self, idx):
        return self.imgs[idx][0]

    def __getitem__(self, index):
        imgs = [self.loader(path) for path in self.imgs[index]]

        # data reinforcement
        if self.transform is not None:
            transformedImgs = self.transform(imgs)
        else:
            transformedImgs = imgs
        # convert to tensor
        return [[transforms.ToTensor()(img) for img in transformedImgs], self.imgNames[index]]

    def __len__(self):
        return len(self.imgs)
