import os
import numpy as np
import torch
import pandas as pd
import glob
import torchvision.transforms as T
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


class RMFDMaskToMask(Dataset):
    def __init__(self, data_folder, transform=None):
        self.path = data_folder
        
        self.data_root = os.path.join(
            self.path, 'AFDB_masked_face_dataset/')
            
        self.inference_list = pd.read_csv(
            os.path.join(self.path, "inference_list_m2m.csv"))
        self.inference_list = self.inference_list.to_numpy().squeeze()
        
        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "mask-to-mask_pairs.csv")).to_numpy()
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform

    def __getitem__(self, i):
        label = self.inference_list[i]
        img_path = os.path.join(self.data_root, label)
        img = self.transform(Image.open(img_path).convert("RGB"))
        return img, str(label)

    def __len__(self):
        return len(self.inference_list)
    
    
class RMFDMaskToNonMask(Dataset):
    def __init__(self, data_folder, transform=None):
        self.path = data_folder
        
        self.mask_root = os.path.join(
            self.path, 'AFDB_masked_face_dataset/')
        
        self.nonmask_root = os.path.join(
            self.path, 'AFDB_face_dataset/')
            
        mask_inference_list = pd.read_csv(
            os.path.join(self.path, "inference_list_m2nm_mask.csv"))
        mask_inference_list['path'] = self.mask_root + mask_inference_list['path'].astype(str)
        
        nonmask_inference_list = pd.read_csv(
            os.path.join(self.path, "inference_list_m2nm_nonmask.csv"))
        nonmask_inference_list['path'] = self.nonmask_root + nonmask_inference_list['path'].astype(str)
        
        self.inference_list = pd.concat((mask_inference_list, 
                                         nonmask_inference_list))
        self.inference_list = self.inference_list.to_numpy().squeeze()
        
        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "mask-to-nonmask_pairs.csv")).to_numpy()
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform

    def __getitem__(self, i):
        img_path = self.inference_list[i]
        img = self.transform(Image.open(img_path).convert("RGB"))
        label = os.path.join(*Path(img_path).parts[-2:])
        return img, str(label)

    def __len__(self):
        return len(self.inference_list)
    
    
class RMFDNonMask2NonMask(Dataset):
    def __init__(self, data_folder, transform=None):
        self.path = data_folder
        
        self.data_root = os.path.join(
            self.path, 'AFDB_face_dataset/')
            
        self.inference_list = pd.read_csv(
            os.path.join(self.path, "inference_list_nm2nm.csv"))
        self.inference_list = self.inference_list.to_numpy().squeeze()
        
        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "nonmask-to-nonmask_pairs.csv")).to_numpy()
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform

    def __getitem__(self, i):
        label = self.inference_list[i]
        img_path = os.path.join(self.data_root, label)
        img = self.transform(Image.open(img_path).convert("RGB"))
        return img, str(label)

    def __len__(self):
        return len(self.inference_list)
    
    
class RMFDMaskToMaskAligned(Dataset):
    def __init__(self, data_folder, transform=None):
        self.path = data_folder
        
        self.data_root = os.path.join(
            self.path, 'AFDB_masked_face_dataset_aligned/')
            
        self.inference_list = pd.read_csv(
            os.path.join(self.path, "inference_list_m2m.csv"))
        self.inference_list = self.inference_list.to_numpy().squeeze()
        
        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "mask-to-mask_pairs.csv")).to_numpy()
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform

    def __getitem__(self, i):
        label = self.inference_list[i]
        img_path = os.path.join(self.data_root, label)
        img = self.transform(Image.open(img_path).convert("RGB"))
        return img, str(label)

    def __len__(self):
        return len(self.inference_list)
    
    
class RMFDMaskToNonMaskAligned(Dataset):
    def __init__(self, data_folder, transform=None):
        self.path = data_folder
        
        self.mask_root = os.path.join(
            self.path, 'AFDB_masked_face_dataset_aligned/')
        
        self.nonmask_root = os.path.join(
            self.path, 'AFDB_face_dataset_aligned/')
            
        mask_inference_list = pd.read_csv(
            os.path.join(self.path, "inference_list_m2nm_mask.csv"))
        mask_inference_list['path'] = self.mask_root + mask_inference_list['path'].astype(str)
        
        nonmask_inference_list = pd.read_csv(
            os.path.join(self.path, "inference_list_m2nm_nonmask.csv"))
        nonmask_inference_list['path'] = self.nonmask_root + nonmask_inference_list['path'].astype(str)
        
        self.inference_list = pd.concat((mask_inference_list, 
                                         nonmask_inference_list))
        self.inference_list = self.inference_list.to_numpy().squeeze()
        
        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "mask-to-nonmask_pairs.csv")).to_numpy()
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform

    def __getitem__(self, i):
        img_path = self.inference_list[i]
        img = self.transform(Image.open(img_path).convert("RGB"))
        label = os.path.join(*Path(img_path).parts[-2:])
        return img, str(label)

    def __len__(self):
        return len(self.inference_list)
    
    
class RMFDNonMask2NonMaskAligned(Dataset):
    def __init__(self, data_folder, transform=None):
        self.path = data_folder
        
        self.data_root = os.path.join(
            self.path, 'AFDB_face_dataset_aligned/')
            
        self.inference_list = pd.read_csv(
            os.path.join(self.path, "inference_list_nm2nm.csv"))
        self.inference_list = self.inference_list.to_numpy().squeeze()
        
        self.pairs_file = pd.read_csv(
            os.path.join(self.path, "nonmask-to-nonmask_pairs.csv")).to_numpy()
        
        if transform is None:
            self.transform = T.Compose([
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform

    def __getitem__(self, i):
        label = self.inference_list[i]
        img_path = os.path.join(self.data_root, label)
        img = self.transform(Image.open(img_path).convert("RGB"))
        return img, str(label)

    def __len__(self):
        return len(self.inference_list)