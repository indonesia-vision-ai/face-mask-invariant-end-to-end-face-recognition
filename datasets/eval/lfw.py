import os
import bcolz
import numpy as np
import torch
import pandas as pd
import glob
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class LFWAligned112Bcolz(Dataset):
    def __init__(self, data_folder):
        self.path = data_folder

        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "lfw_pair_list.tsv"),
            names=['p1','p2','gt'], sep='\t').to_numpy()

        self.images = bcolz.carray(
            rootdir=os.path.join(self.path, "lfw"), mode='r')

    def __getitem__(self, i):
        image = self.images[i]
        return torch.tensor(image), str(i)

    def __len__(self):
        return len(self.images)


class LFWRaw(Dataset):
    r"""LFW dataset with original unmodified images

    Args:
        data_folder (str): Root directory for the dataset files
        crop (str, optional): Whether to use ulfg cropped images
        transform (callable, optional): A function/transform that
            takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """
    def __init__(self, data_folder, transform=None):
        self.path = data_folder

        self.data_root = os.path.join(
            self.path, 'lfw')

        self.images = glob.glob(os.path.join(
            self.data_root, '*', '*.jpg'))

        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "lfw_test_pair.csv")).to_numpy()

        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform

    def __getitem__(self, i):
        img = self.transform(Image.open(self.images[i]))
        label = os.path.join(*Path(self.images[i]).parts[-2:])
        return img, str(label)

    def __len__(self):
        return len(self.images)


class LFWMasked(LFWRaw):
    r"""LFW dataset with facemask augmented images

    Args:
        data_folder (str): Root directory for the dataset files
        transform (callable, optional): A function/transform that
            takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """
    def __init__(self, data_folder, transform=None):
        self.path = data_folder

        self.data_root = os.path.join(
            self.path, 'lfw_masked')

        self.images = glob.glob(os.path.join(
            self.data_root, '*', '*.jpg'))

        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "lfw_test_pair.csv")).to_numpy()

        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform


class LFWOriVsMasked(LFWRaw):
    r"""LFW dataset with both facemask augmented and original images,
    where the match is face without mask vs with mask

    Args:
        data_folder (str): Root directory for the dataset files
        transform (callable, optional): A function/transform that
            takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """
    def __init__(self, data_folder, transform=None):
        self.path = data_folder

        self.ori_root = os.path.join(self.path, 'lfw')
        self.masked_root = os.path.join(self.path, 'lfw_masked')

        self.ori_images = glob.glob(os.path.join(
            self.ori_root, '*', '*.jpg'))
        self.masked_images = glob.glob(os.path.join(
            self.masked_root, '*', '*.jpg'))
        self.images = self.ori_images + self.masked_images

        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "lfw_ori_v_mask_pair.csv")
        ).to_numpy()

        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform

    def __getitem__(self, i):
        img = self.transform(Image.open(self.images[i]))
        label = os.path.join(*Path(self.images[i]).parts[-3:])
        return img, str(label)


class LFWMaskedAlign112(LFWRaw):
    r"""LFW dataset with facemask augmented images

    Args:
        data_folder (str): Root directory for the dataset files
        transform (callable, optional): A function/transform that
            takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """
    def __init__(self, data_folder, transform=None):
        self.path = data_folder

        self.data_root = os.path.join(
            self.path, 'lfw_masked_align_naive')

        self.images = glob.glob(os.path.join(
            self.data_root, '*', '*.jpg'))

        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "lfw_test_pair.csv")).to_numpy()

        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform


class LFWMaskedAlignFromOri112(LFWRaw):
    r"""LFW dataset with facemask augmented images

    Args:
        data_folder (str): Root directory for the dataset files
        transform (callable, optional): A function/transform that
            takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """
    def __init__(self, data_folder, transform=None):
        self.path = data_folder

        self.data_root = os.path.join(
            self.path, 'lfw_masked_align_fromori')

        self.images = glob.glob(os.path.join(
            self.data_root, '*', '*.jpg'))

        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "lfw_test_pair.csv")).to_numpy()

        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform


class LFWAligned112(Dataset):
    r"""LFW dataset with original unmodified images, aligned and cropped to 112

    Args:
        data_folder (str): Root directory for the dataset files
        crop (str, optional): Whether to use ulfg cropped images
        transform (callable, optional): A function/transform that
            takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """
    def __init__(self, data_folder, transform=None):
        self.path = data_folder

        self.data_root = os.path.join(
            self.path, 'lfw_aligned')

        self.images = glob.glob(os.path.join(
            self.data_root, '*', '*.jpg'))

        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "lfw_test_pair.csv")).to_numpy()

        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform

    def __getitem__(self, i):
        img = self.transform(Image.open(self.images[i]))
        label = os.path.join(*Path(self.images[i]).parts[-2:])
        return img, str(label)

    def __len__(self):
        return len(self.images)


class LFWOriVsMaskedAligned112(LFWRaw):
    r"""LFW dataset with both facemask augmented and original images,
    where the match is face without mask vs with mask

    Args:
        data_folder (str): Root directory for the dataset files
        transform (callable, optional): A function/transform that
            takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """
    def __init__(self, data_folder, masked_align_type, transform=None):
        assert masked_align_type in ["naive", "fromori"]
        self.path = data_folder

        self.ori_root = os.path.join(self.path, 'lfw_aligned')

        if masked_align_type == "naive":
            masked_root_name = 'lfw_masked_align_naive'
        else:
            masked_root_name = 'lfw_masked_align_fromori'

        self.masked_root = os.path.join(self.path, masked_root_name)

        self.ori_images = glob.glob(os.path.join(
            self.ori_root, '*', '*.jpg'))
        self.masked_images = glob.glob(os.path.join(
            self.masked_root, '*', '*.jpg'))
        self.images = self.ori_images + self.masked_images

        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "lfw_ori_v_mask_pair.csv")
        )

        self.pairs_file['p1'] = self.pairs_file['p1'].str.replace('lfw/','lfw_aligned/')
        self.pairs_file['p2'] = self.pairs_file['p2'].str.replace('lfw_masked/', masked_root_name + '/')

        print(self.pairs_file)

        self.pairs_file = self.pairs_file.to_numpy()

        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform

    def __getitem__(self, i):
        img = self.transform(Image.open(self.images[i]))
        label = os.path.join(*Path(self.images[i]).parts[-3:])
        return img, str(label)