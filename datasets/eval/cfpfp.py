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


class CFPFPAlign112(Dataset):
    def __init__(self, data_folder):
        self.path = data_folder

        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "cfp_fp_pair_list.tsv"),
            names=['p1','p2','gt'], sep='\t').to_numpy()

        self.images = bcolz.carray(
            rootdir=os.path.join(self.path, "cfp_fp"), mode='r')

    def __getitem__(self, i):
        image = self.images[i]
        return torch.tensor(image), str(i)

    def __len__(self):
        return len(self.images)
    
    
# ==============================================================================

    
class CFPFPRaw(Dataset):
    r"""CFPFP dataset with original unmodified images

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
            self.path, 'Data', 'Images')
            
        self.images = glob.glob(os.path.join(
            self.data_root, '**', '**', '*.jpg'))
        
        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "Protocol", "cfp_fp_pairs.csv")).to_numpy()
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform

    def __getitem__(self, i):
        img = self.transform(Image.open(self.images[i]).convert('RGB'))
        label = os.path.join(*Path(self.images[i]).parts[-3:])
        return img, str(label)

    def __len__(self):
        return len(self.images)


class CFPFPMasked(CFPFPRaw):
    r"""CFPFP dataset with facemask augmented images

    Args:
        data_folder (str): Root directory for the dataset files
        transform (callable, optional): A function/transform that 
            takes in an PIL image and returns a transformed version. 
            E.g, ``transforms.RandomCrop``
    """
    def __init__(self, data_folder, transform=None):
        self.path = data_folder
        
        self.data_root = os.path.join(
            self.path, 'Data', 'Images_frontal_masked')
            
        self.images = glob.glob(os.path.join(
            self.data_root, '**', '**', '*.jpg'))
        
        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "Protocol", "cfp_fp_pairs.csv")).to_numpy()
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform
            
            
class CFPFPMask2Raw(Dataset):
    def __init__(self, data_folder, transform=None):
        self.path = data_folder
        
        raw_data_root = os.path.join(self.path, 'Data', 'Images/')
        mask_data_root = os.path.join(self.path, 'Data', 'Images_frontal_masked/')

        self.pairs_file = pd.read_csv(os.path.join(self.path, "Protocol", "cfp_fp_pairs.csv"))
        
        p1_images = pd.DataFrame({'path': self.pairs_file.p1})
        p1_images['path'] = mask_data_root + p1_images['path']
        
        p2_images = pd.DataFrame({'path': self.pairs_file.p2})
        p2_images['path'] = raw_data_root + p2_images['path']
        
        self.images = pd.concat((p1_images, p2_images)).to_numpy().squeeze()
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
        img = self.transform(Image.open(self.images[i]).convert('RGB'))
        label = os.path.join(*Path(self.images[i]).parts[-3:])
        return img, str(label)

    def __len__(self):
        return len(self.images)
            
            
# ==============================================================================
            

class CFPFPMaskedAligned(CFPFPRaw):
    r"""CFPFP dataset with facemask augmented images

    Args:
        data_folder (str): Root directory for the dataset files
        transform (callable, optional): A function/transform that 
            takes in an PIL image and returns a transformed version. 
            E.g, ``transforms.RandomCrop``
    """
    def __init__(self, data_folder, transform=None):
        self.path = data_folder
        
        self.data_root = os.path.join(
            self.path, 'Data', 'Images_frontal_masked_aligned')
            
        self.images = glob.glob(os.path.join(
            self.data_root, '**', '**', '*.jpg'))
        
        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "Protocol", "cfp_fp_pairs.csv")).to_numpy()
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform
            
            
class CFPFPMask2RawAligned(Dataset):
    def __init__(self, data_folder, transform=None):
        self.path = data_folder
        
        raw_data_root = os.path.join(self.path, 'Data', 'Images_aligned/')
        mask_data_root = os.path.join(self.path, 'Data', 'Images_frontal_masked_aligned/')

        self.pairs_file = pd.read_csv(os.path.join(self.path, "Protocol", "cfp_fp_pairs.csv"))
        
        p1_images = pd.DataFrame({'path': self.pairs_file.p1})
        p1_images['path'] = mask_data_root + p1_images['path']
        
        p2_images = pd.DataFrame({'path': self.pairs_file.p2})
        p2_images['path'] = raw_data_root + p2_images['path']
        
        self.images = pd.concat((p1_images, p2_images)).to_numpy().squeeze()
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
        img = self.transform(Image.open(self.images[i]).convert('RGB'))
        label = os.path.join(*Path(self.images[i]).parts[-3:])
        return img, str(label)

    def __len__(self):
        return len(self.images)