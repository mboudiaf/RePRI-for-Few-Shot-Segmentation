import cv2
import numpy as np
import src.dataset.transform as transform
from torch.utils.data import Dataset
from .utils import make_dataset
from .classes import get_split_classes, filter_classes
import torch
import random
import argparse
from typing import List
from torch.utils.data.distributed import DistributedSampler


def get_train_loader(args: argparse.Namespace,
                     return_paths: bool = False) -> torch.utils.data.DataLoader:
    """
        Build the train loader. This is a standard loader (not episodic)
    """
    assert args.train_split in [0, 1, 2, 3]
    aug_dic = {'randscale': transform.RandScale([args.scale_min, args.scale_max]),
               'randrotate': transform.RandRotate([args.rot_min, args.rot_max],
                                                  padding=[0 for x in args.mean],
                                                  ignore_label=255),
               'hor_flip': transform.RandomHorizontalFlip(),
               'vert_flip': transform.RandomVerticalFlip(),
               'crop': transform.Crop([args.image_size, args.image_size], crop_type='rand',
                                      padding=[0 for x in args.mean], ignore_label=255),
               'resize': transform.Resize(args.image_size)
               }

    train_transform = [aug_dic[name] for name in args.augmentations]
    train_transform += [transform.ToTensor(), transform.Normalize(mean=args.mean, std=args.std)]
    train_transform = transform.Compose(train_transform)

    split_classes = get_split_classes(args)
    class_list = split_classes[args.train_name][args.train_split]['train']

    # ===================== Build loader =====================
    train_data = StandardData(transform=train_transform,
                              class_list=class_list,
                              return_paths=return_paths,
                              data_list_path=args.train_list,
                              args=args)

    world_size = torch.distributed.get_world_size()
    train_sampler = DistributedSampler(train_data) if args.distributed else None
    batch_size = int(args.batch_size / world_size) if args.distributed else args.batch_size

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)
    return train_loader, train_sampler


def get_val_loader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    """
        Build the episodic validation loader.
    """
    assert args.test_split in [0, 1, 2, 3, -1, 'default']
    val_transform = transform.Compose([
            transform.Resize(args.image_size),
            transform.ToTensor(),
            transform.Normalize(mean=args.mean, std=args.std)])
    split_classes = get_split_classes(args)

    # ===================== Filter out classes seen during training =====================
    if args.test_name == 'default':
        test_name = args.train_name
        test_split = args.train_split
    else:
        test_name = args.test_name
        test_split = args.test_split
    class_list = filter_classes(args.train_name, args.train_split, test_name, test_split, split_classes)

    # ===================== Build loader =====================
    if args.episodic_val:
        val_sampler = None
        val_data = EpisodicData(transform=val_transform,
                                class_list=class_list,
                                data_list_path=args.val_list,
                                args=args)
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True,
                                                 sampler=val_sampler)
    else:
        class_list = split_classes[args.train_name][args.train_split]['train']
        val_data = StandardData(args=args,
                                transform=val_transform,
                                class_list=class_list,
                                return_paths=False,
                                data_list_path=args.val_list)
        val_sampler = DistributedSampler(val_data) if args.distributed else None
        world_size = torch.distributed.get_world_size()
        batch_size = int(args.batch_size_val / world_size) if args.distributed else args.batch_size_val
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=1,
                                                 pin_memory=True,
                                                 sampler=val_sampler)

    return val_loader, val_transform


class StandardData(Dataset):
    def __init__(self, args: argparse.Namespace,
                 transform: transform.Compose,
                 data_list_path: str,
                 class_list: List[int],
                 return_paths: bool):
        self.data_root = args.data_root
        self.class_list = class_list
        self.data_list, _ = make_dataset(args.data_root, data_list_path, class_list)
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        label_class = []
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        new_label_class = []
        undesired_class = []
        for c in label_class:
            if c in self.class_list:
                new_label_class.append(c)
            else:
                undesired_class.append(c)
        label_class = new_label_class
        assert len(label_class) > 0

        new_label = np.zeros_like(label)  # background
        for lab in label_class:
            indexes = np.where(label == lab)
            new_label[indexes[0], indexes[1]] = self.class_list.index(lab) + 1  # Add 1 because class 0 is for bg
        for lab in undesired_class:
            indexes = np.where(label == lab)
            new_label[indexes[0], indexes[1]] = 255

        ignore_pix = np.where(new_label == 255)
        new_label[ignore_pix[0], ignore_pix[1]] = 255

        if self.transform is not None:
            image, new_label = self.transform(image, new_label)
        if self.return_paths:
            return image, new_label, image_path, label_path
        else:
            return image, new_label


class EpisodicData(Dataset):
    def __init__(self,
                 transform: transform.Compose,
                 class_list: List[int],
                 data_list_path: str,
                 args: argparse.Namespace):

        self.shot = args.shot
        self.random_shot = args.random_shot
        self.data_root = args.data_root
        self.class_list = class_list
        self.data_list, self.sub_class_file_list = make_dataset(args.data_root, data_list_path, self.class_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        # ========= Read query image + Chose class =========================
        label_class = []
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        new_label_class = []
        for c in label_class:
            if c in self.class_list:  # current list of classes to try
                new_label_class.append(c)
        label_class = new_label_class
        assert len(label_class) > 0

        # == From classes in query image, chose one randomly ===

        class_chosen = np.random.choice(label_class)
        new_label = np.zeros_like(label)
        ignore_pix = np.where(label == 255)
        target_pix = np.where(label == class_chosen)
        new_label[ignore_pix] = 255
        new_label[target_pix] = 1
        label = new_label

        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)

        # ========= Build support ==============================================

        # == First, randomly choose indexes of support images  =
        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []

        if self.random_shot:
            shot = random.randint(1, self.shot)
        else:
            shot = self.shot

        for k in range(shot):
            support_idx = random.randint(1, num_file) - 1
            support_image_path = image_path
            support_label_path = label_path
            while((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1, num_file) - 1
                support_image_path, support_label_path = file_class_chosen[support_idx]
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list = []
        support_label_list = []
        subcls_list = [self.class_list.index(class_chosen) + 1]

        # == Second, read support images and masks  ============
        for k in range(shot):
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k]
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:, :] = 0
            support_label[target_pix[0], target_pix[1]] = 1
            support_label[ignore_pix[0], ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))
            support_image_list.append(support_image)
            support_label_list.append(support_label)
        assert len(support_label_list) == shot and len(support_image_list) == shot

        # == Forward images through transforms =================
        if self.transform is not None:
            qry_img, target = self.transform(image, label)
            for k in range(shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list[k], support_label_list[k])
                support_image_list[k] = support_image_list[k].unsqueeze(0)
                support_label_list[k] = support_label_list[k].unsqueeze(0)

        # == Reshape properly ==================================
        spprt_imgs = torch.cat(support_image_list, 0)
        spprt_labels = torch.cat(support_label_list, 0)

        return qry_img, target, spprt_imgs, spprt_labels, subcls_list, support_image_path_list, [image_path]
