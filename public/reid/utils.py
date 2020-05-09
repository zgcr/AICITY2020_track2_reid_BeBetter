import os
import sys
import warnings

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import cv2
import copy
import math
import numpy as np
import random
import torch
import pickle as pkl
import logging
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from logging.handlers import TimedRotatingFileHandler
import torchvision.transforms as transforms
from torch.utils.data import Sampler
from torch.utils.data import Dataset


class VehicleTrainDataset(Dataset):
    def __init__(self,
                 root_path,
                 train_dataset_pkl,
                 input_size=(320, 320),
                 use_crop_box=True,
                 use_background_substitution=False,
                 transform=None):
        self.root_path = root_path
        self.train_dataset_pkl = train_dataset_pkl
        self.metas = []
        for dataset in tqdm(self.train_dataset_pkl):
            with open(dataset, 'rb') as f:
                items = pkl.load(f, encoding='bytes')
                self.metas.extend(items)
        self.metas, _, _ = self.relabel(self.metas)

        self.input_size = input_size
        self.transform = transform
        self.use_crop_box = use_crop_box
        self.use_background_substitution = use_background_substitution

    def relabel(self, metas):
        raw_ids = set()
        metas = metas.copy()
        for item in metas:
            raw_ids.add(item["vehicle_id"])
        raw_ids = sorted(list(raw_ids))
        rawid2label = {raw_vid: i for i, raw_vid in enumerate(raw_ids)}
        label2rawid = {i: raw_vid for i, raw_vid in enumerate(raw_ids)}
        for item in metas:
            item["vehicle_id"] = rawid2label[item["vehicle_id"]]

        return metas, rawid2label, label2rawid

    def crop_box(self, sample, meta):
        h, w, _ = sample["image"].shape
        if meta['bbox']:
            lx, ly, rx, ry, _ = list(map(int, meta['bbox']))
            if ((lx >= 0) and (ly >= 0) and (rx <= w) and
                (ry <= h)) and (lx < rx - 20 and ly < ry - 20):
                sample["image"] = sample["image"][ly:ry, lx:rx, :]

        return sample

    def background_substitution(self, sample, meta, prob=0.5, mask_num=5):
        sample["mask"] = meta['mask']
        origin_h, origin_w = sample["image_shape"]
        sample["mask"] = [sample["mask"] == v for v in range(mask_num)]
        sample["mask"] = np.stack(sample["mask"], axis=-1).astype('float32')
        sample["mask"] = cv2.resize(sample["mask"], (origin_w, origin_h),
                                    0,
                                    0,
                                    interpolation=cv2.INTER_NEAREST)
        # crop mask
        if self.crop_box and meta['bbox']:
            lx, ly, rx, ry, _ = list(map(int, meta['bbox']))
            if ((lx >= 0) and (ly >= 0) and (rx <= origin_w) and
                (ry <= origin_h)) and (lx < rx - 20 and ly < ry - 20):
                sample["mask"] = sample["mask"][ly:ry, lx:rx]

        if np.random.rand() < prob:
            image = sample["image"]
            h, w, _ = image.shape
            mask = sample["mask"][:, :, 0] > 0.5
            mask = mask.reshape(h, w, 1)
            bg = self.get_empty_background()
            bg = cv2.resize(bg, (w, h), 0, 0)
            fg = image * (1 - mask)
            bg = bg * mask
            sample["image"] = bg + fg

        sample['image'] = sample['image'].astype("uint8")

        del sample["mask"]

        return sample

    def read_mask(self, sample, mask_num=5):
        mask = sample['mask']
        mask = [mask == v for v in range(mask_num)]
        h, w, _ = sample['image'].shape
        mask = np.stack(mask, axis=-1).astype('float32')
        mask = cv2.resize(mask, (w, h), 0, 0, interpolation=cv2.INTER_NEAREST)
        sample["mask"] = mask

        return sample

    def get_empty_background(self):
        idx = np.random.randint(0, len(self.metas))
        meta = self.metas[idx]
        bg_sample = meta.copy()
        image_path = os.path.join(self.root_path, bg_sample["image_path"])
        bg_sample["image"] = cv2.imread(image_path)
        bg_sample = self.read_mask(bg_sample)
        H, W, _ = bg_sample['mask'].shape
        mask = bg_sample['mask'][:, :, 0] > 0.5
        bg = bg_sample['image'] * mask.reshape(H, W, 1)
        bg_avg = bg[mask != 0, :].mean(axis=0)
        bg_fill = bg_avg.reshape(1, 1, 3) * (1 - mask).reshape(H, W, 1)
        bg = bg + bg_fill

        return bg

    def __getitem__(self, idx):
        sample = {}
        image_path = os.path.join(self.root_path,
                                  self.metas[idx]["image_path"])
        sample["image"] = cv2.imread(image_path)
        sample["vehicle_id"] = self.metas[idx]["vehicle_id"]
        origin_h, origin_w, _ = sample['image'].shape
        sample["image_shape"] = [origin_h, origin_w]

        if self.use_crop_box:
            sample = self.crop_box(sample, self.metas[idx])

        if self.use_background_substitution:
            sample = self.background_substitution(sample, self.metas[idx])

        del sample["image_shape"]

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

    def __len__(self):
        return len(self.metas)


class VehicleValDataset(Dataset):
    def __init__(self,
                 root_path,
                 val_dataset_pkl,
                 input_size=(320, 320),
                 use_crop_box=True,
                 transform=None):
        self.root_path = root_path
        with open(val_dataset_pkl, 'rb') as f:
            self.val_dataset = pkl.load(f, encoding='bytes')
        self.query_meta = self.val_dataset['reid_query']
        self.gallery_meta = self.val_dataset['reid_gallery']
        self.total_meta = self.query_meta + self.gallery_meta
        self.input_size = input_size
        self.use_crop_box = use_crop_box
        self.transform = transform

    def crop_box(self, image, meta):
        h, w, _ = image.shape
        if meta['bbox']:
            lx, ly, rx, ry, _ = list(map(int, meta['bbox']))
            if ((lx >= 0) and (ly >= 0) and (rx <= w) and
                (ry <= h)) and (lx < rx - 20 and ly < ry - 20):
                image = image[ly:ry, lx:rx, :]

        return image

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_path,
                                  self.total_meta[idx]["image_path"])
        image = cv2.imread(image_path)

        if self.use_crop_box:
            image = self.crop_box(image, self.total_meta[idx])

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.total_meta)


class RandomShrink(object):
    def __init__(self,
                 prob=0.5,
                 input_size=320,
                 interpolation=cv2.INTER_LINEAR):
        super(RandomShrink, self).__init__()
        self.prob = 0.5
        self.input_size = input_size
        self.interpolation = interpolation

    def __call__(self, img):
        height, width, _ = img.shape
        size = (width, height)
        if np.random.rand() < self.prob:
            scale = random.random() * 0.4 + 0.2
            scale_size = (int(width * scale), int(height * scale))
            if width > self.input_size or height > self.input_size:
                img = cv2.resize(img, scale_size, self.interpolation)
            return cv2.resize(img, size, interpolation=self.interpolation)
        else:
            return img


class RandomErasing(object):
    """Randomly selects a rectangle region in an image and erases its pixels.
       'Random Erasing Data Augmentation' by Zhong et al.
       See https://arxiv.org/pdf/1708.04896.pdf

    :param float probability: The probability that the Random Erasing operation will be performed.
    :param float sl: Minimum proportion of erased area against input image.
    :param float sh: Maximum proportion of erased area against input image.
    :param float r1: Minimum aspect ratio of erased area.
    :param tuple mean: Erasing value.
    """
    def __init__(self,
                 probability=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=(0.4465, 0.4822, 0.4914)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def handle_numpy(self, img):
        """
        img为未归一化的(H,W,C)，为albumentation使用
        :param img:
        :return:
        """
        shape = img.shape
        if random.uniform(0, 1) >= self.probability:
            return img

        for _ in range(100):
            area = shape[0] * shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < shape[1] and h < shape[0]:
                x1 = random.randint(0, shape[0] - h)
                y1 = random.randint(0, shape[1] - w)
                if shape[2] == 3:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0] * 255
                    img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1] * 255
                    img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2] * 255
                else:
                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                return img

        return img

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return self.handle_numpy(img)
        else:
            img = np.asarray(img).copy()
            img = self.handle_numpy(img)
            return Image.fromarray(img)

    def __repr__(self):
        return self.__class__.__name__ + '(prob={})'.format(self.probability)


class CoverageKPSampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        super(CoverageKPSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, item in enumerate(self.data_source):
            pid = item["vehicle_id"]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs,
                                        size=self.num_instances,
                                        replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


def get_logger(name, log_dir='log'):
    """
    Args:
        name(str): name of logger
        log_dir(str): path of log
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = TimedRotatingFileHandler(info_name,
                                            when='D',
                                            encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    error_name = os.path.join(log_dir, '{}.error.log'.format(name))
    error_handler = TimedRotatingFileHandler(error_name,
                                             when='D',
                                             encoding='utf-8')
    error_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    info_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    logger.addHandler(info_handler)
    logger.addHandler(error_handler)

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


if __name__ == "__main__":
    root_path = '/data/AICITY2020/AIC20_track2/'
    train_dataset_pkl = [
        '/data/aicity_pkl/track2_train_pytorch.pkl',
        '/data/aicity_pkl/track2_simu_train_pytorch.pkl',
    ]
    val_dataset_pkl = '/data/aicity_pkl/benchmark_pytorch.pkl'

    train = VehicleTrainDataset(root_path,
                                train_dataset_pkl,
                                input_size=(320, 320),
                                use_background_substitution=True,
                                transform=transforms.Compose([
                                    RandomShrink(),
                                    transforms.ToPILImage(),
                                    transforms.Resize((320, 320)),
                                    transforms.Pad(10),
                                    transforms.RandomCrop(320),
                                    RandomErasing(),
                                    transforms.ToTensor(),
                                ]))
    for i in range(10):
        print(train[i])

    val = VehicleValDataset(root_path,
                            val_dataset_pkl,
                            input_size=(320, 320),
                            use_crop_box=True,
                            transform=transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((320, 320)),
                                transforms.ToTensor(),
                            ]))
    for i in range(10):
        print(val[i], val[i].shape)
