from pathlib import Path
import argparse

import numpy as np
import imgaug.augmenters as iaa
import torchvision.transforms as T
import torch

from Evaluator import FRLiteEvaluator
from Predictor import FRLitePredictor
from networks.ResnetFaceSTN import ResnetFaceSTN
from networks.InsightFace import Backbone, MobileFaceNet
from networks.InsightFace2 import ResNet, IRBlock
from datasets.eval.lfw import (LFWMasked, LFWOriVsMasked, LFWRaw,
                               LFWMaskedAlign112, LFWMaskedAlignFromOri112,
                               LFWOriVsMaskedAligned112, LFWAligned112,
                               LFWAligned112Bcolz)
from datasets.eval.cplfw import (CPLFWRaw, CPLFWMasked, CPLFWMaskedAligned,
                                 CPLFWMask2Raw, CPLFWMask2RawAligned)
from datasets.eval.calfw import (CALFWRaw, CALFWMasked, CALFWMaskedAligned,
                                 CALFWMask2Raw, CALFWMask2RawAligned)
from datasets.eval.cfpff import (CFPFFRaw, CFPFFMasked, CFPFFMaskedAligned,
                                 CFPFFMask2Raw, CFPFFMask2RawAligned)
from datasets.eval.cfpfp import (CFPFPRaw, CFPFPMasked, CFPFPMaskedAligned,
                                 CFPFPMask2Raw, CFPFPMask2RawAligned)
from datasets.eval.rmfd import (RMFDMaskToMask, RMFDMaskToNonMask,
                                RMFDNonMask2NonMask, RMFDMaskToMaskAligned,
                                RMFDMaskToNonMaskAligned,
                                RMFDNonMask2NonMaskAligned)


# WEIGHT_PATH = 'weights/InsightFace/SE-LResNet50E-IR.pth'
# WEIGHT_PATH = 'weights/SE-LResNet50E-IR-CASIA_MASK_CLEAN_ALIGNED/epoch_32/SE-LResNet50E-IR-CASIA_MASK_CLEAN_ALIGNED_ep32.pth'
# WEIGHT_PATH = "weights/mask_exp3-resnet-casia_mask/epoch_36/mask_exp3-resnet-casia_mask_ep36.pth"
WEIGHT_PATH = "weights/mask_exp19-resnetSTN/epoch_18/mask_exp19-resnetSTN_ep18.pth"

# SAVE_DIR = "eval_saves/InsightFace2/Arcface (R50-SE, MS1M)"
# SAVE_DIR = 'eval_saves/InsightFace2/Arcface (R50-SE, mCASIA)'
# SAVE_DIR = "eval_saves/STN-IR-SE50 (mCASIA)"
SAVE_DIR = "eval_saves/STN-IR-SE50 (mCASIA, C)"

# SAVE_NAME = "Arcface (R50-SE, MS1M) CFPFPMask2RawAligned"
# SAVE_NAME = "Arcface (R50-SE, mCASIA) CALFWMask2RawAligned"
# SAVE_NAME = 'STN-IR-SE50 (mCASIA) CFPFPMask2Raw'
SAVE_NAME = 'STN-IR-SE50 (mCASIA, C) CPLFWMask2Raw'

# net = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
# net = ResNet(IRBlock, [3, 4, 6, 3], use_se=True)
net = ResnetFaceSTN(stn_mode='resnet')
net.load_state_dict(torch.load(WEIGHT_PATH))

evaluator = FRLiteEvaluator(
    net,
    torch.device('cuda'),
    num_workers=24,
    batch_size=512
)

# class Transform:
#     def __init__(self):
#         self.aug = iaa.Sequential([
#             iaa.CenterPadToSquare(),
#             iaa.CenterPadToPowersOf(width_base=2, height_base=2),
#             iaa.Resize({"height": 128, "width": 128})
#         ])

#     def __call__(self, img):
#         img = np.asarray(img)
#         return self.aug.augment_image(img)

eval_transform = T.Compose([
    T.Resize((128, 128)),
    # Transform(),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# eval_dataset = LFWOriVsMaskedAligned112('datasets/eval/data/lfw', 'fromori', transform=eval_transform)
# eval_dataset = LFWAligned112Bcolz('datasets/eval/data/lfw_align_112')
eval_dataset = CPLFWMask2Raw(
    'datasets/eval/data/cplfw', transform=eval_transform)

evaluator.evaluate(eval_dataset, SAVE_DIR, SAVE_NAME)
