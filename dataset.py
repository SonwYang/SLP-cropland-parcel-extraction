from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torchvision
from torchvision.transforms import Compose
import numpy as np
import cv2 as cv
import os
from random import sample
from torchvision.transforms import functional as F


def img_to_tensor(img):
    tensor = torch.from_numpy(img.transpose((2, 0, 1)))
    return tensor


def to_monochrome(x):
    # x_ = x.convert('L')
    x_ = np.array(x).astype(np.float32)  # convert image to monochrome
    return x_


def to_tensor(x):
    x_ = np.expand_dims(x, axis=0)
    x_ = torch.from_numpy(x_)
    return x_


ImageToTensor = torchvision.transforms.ToTensor


def custom_blur_demo(image):
  kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
  dst = cv.filter2D(image, -1, kernel=kernel)
  return dst


class SasDataset(Dataset):
    def __init__(self, root, mode='train', is_ndvi=False):
        self.root = root
        self.mode = mode
        self.mean_bgr = [104.00699, 116.66877, 122.67892]
        self.is_ndvi = is_ndvi
        self.imgList = sorted(img for img in os.listdir(self.root))
        self.imgTransforms = Compose([img_to_tensor])
        self.maskTransforms = Compose([
            torchvision.transforms.Lambda(to_monochrome),
            torchvision.transforms.Lambda(to_tensor),
        ])

    def __getitem__(self, idx):
        imgPath = os.path.join(self.root, self.imgList[idx])
        img = cv.imread(imgPath, cv.IMREAD_COLOR)
        img = np.array(img, dtype=np.float32)
        # if self.rgb:
        #     img = img[:, :, ::-1]  # RGB->BGR
        img /= 255.
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        imgName = os.path.split(imgPath)[-1].split('.')[0]

        if self.mode == 'test':
            batch_data = {'img': img, 'file_name': imgName}
            return batch_data

    def __len__(self):
        return len(self.imgList)


def build_loader(cfg):
    # Get correct indices
    num_train = len(sorted(img for img in os.listdir(cfg.trainData)))
    indices = list(range(num_train))
    indices = sample(indices, len(indices))
    split = int(np.floor(0.15 * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # set sup datasets
    train_dataset = SasDataset(cfg.trainData, mode='train')
    val_dataset = SasDataset(cfg.trainData, mode='valid')

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=valid_sampler,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    return train_loader, valid_loader


if __name__=='__main__':
    import matplotlib.pyplot as plt
    from config_eval import Config
    cfg = Config()
    train_loader, valid_loader = build_loader(cfg)

    for x, y in train_loader:
        x = x.numpy() * 255
        y = y.numpy()
        plt.subplot(121)
        plt.imshow(x)
        plt.subplot(122)
        plt.imshow(y)
        plt.show()

    # DECAY_POWER = 3
    # SHAPE = 512
    # LAMBDA = 0.5
    # NUM_IMAGES = 1
    # dataIter = iter(valid_loader)
    # batch, target = next(dataIter)
    # batch1 = batch[:NUM_IMAGES]
    # batch2 = batch[NUM_IMAGES:]
    #
    # soft_masks_np = [make_low_freq_image(DECAY_POWER, [SHAPE, SHAPE]) for _ in range(NUM_IMAGES)]
    # soft_masks = torch.from_numpy(np.stack(soft_masks_np, axis=0)).float().repeat(1, 3, 1, 1)
    #
    # masks_np = [binarise_mask(mask, LAMBDA, [SHAPE, SHAPE]) for mask in soft_masks_np]
    # masks = torch.from_numpy(np.stack(masks_np, axis=0)).float().repeat(1, 3, 1, 1)
    #
    # mix = batch1 * masks + batch2 * (1 - masks)
    # image = torch.cat((soft_masks, masks, batch1, batch2, mix), 0)
    # save_image(image, 'fmix_example.png', nrow=NUM_IMAGES, pad_value=1)
    #
    # plt.figure(figsize=(NUM_IMAGES, 5))
    # plt.imshow(make_grid(image, nrow=NUM_IMAGES, pad_value=5).permute(1, 2, 0).numpy())
    # plt.show()

