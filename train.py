import torch
import numpy as np
import imgaug.augmenters as iaa
import torchvision.transforms as T
import torchvision.datasets as D
from networks.ResnetFaceSTN import ResnetFaceSTN
from networks.head import ArcFaceAlt, LiArcFace
from networks.loss import FocalLoss
from TrainerMaskedPlateau import FRTrainerPlateau

torch.backends.cudnn.benchmark = True


def separate_bn_params(net):
    stn_modules = [*net.localization.modules()] + [*net.fc_loc.modules()]
    all_modules = [*net.modules()]
    recog_modules = [m for m in all_modules if m not in stn_modules]

    params_only_bn = []
    params_wo_bn = []
    params_prelu = []
    for layer in recog_modules:
        if 'networks.ResnetFaceSTN' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                params_only_bn.extend([*layer.parameters()])
            elif 'PReLU' in str(layer.__class__):
                params_prelu.extend([*layer.parameters()])
            else:
                params_wo_bn.extend([*layer.parameters()])

    params_stn = []
    params_stn_only_bn = []
    for layer in stn_modules:
        if 'networks.ResnetFaceSTN' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        if 'batchnorm' in str(layer.__class__):
            params_stn_only_bn.extend([*layer.parameters()])
        else:
            params_stn.extend([*layer.parameters()])

    return (params_only_bn, params_wo_bn, params_prelu,
            params_stn, params_stn_only_bn)


def get_optim_params(net, head, params_config):
    (params_only_bn, params_wo_bn,
     params_prelu, params_stn, params_stn_only_bn) = separate_bn_params(net)

    optim_params = [
        {'params': params_wo_bn + params_prelu + [head.weight],
         'weight_decay': params_config['weight_decay']},
        {'params': params_only_bn},
        {'params': params_stn, 'lr': 5e-4,
         'weight_decay': params_config['weight_decay']},
        {'params': params_stn_only_bn, 'lr': 5e-4}
    ]

    return optim_params


TRAIN_DATASET_DIR = 'datasets/train/data/CASIA_Webface_MaskCombined'

# Transform using imgaug


class Transform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                order=[0, 1],
                mode='edge'
            ),
            iaa.Resize({"height": 128, "width": 128})
        ])

    def __call__(self, img):
        img = np.asarray(img)
        return self.aug.augment_image(img)


train_transform = T.Compose([
    Transform(),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

eval_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = D.ImageFolder(
    TRAIN_DATASET_DIR,
    transform=train_transform
)

print('Num classes:', len(train_dataset.classes))
print('Num images:', len(train_dataset))

default_param_config = {
    'name': 'mask_exp19-resnetSTN',
    'device': 'cuda',
    'multi_gpu': True,
    'num_of_gpu': '4',
    'workers': 16,
    'epochs': 50,
    'batch_size': 512,
    'lr': 1e-1,
    'weight_decay': 5e-4,
    'momentum': 0.9
}

default_net_config = {
    'net': ResnetFaceSTN(stn_mode='resnet'),
    'head': LiArcFace(512, len(train_dataset.classes)),
    'criterion': FocalLoss()
}

default_dataset_config = {
    'train_dataset': train_dataset
}

scheduler_params = {
    'factor': 0.1,
    'patience': 5,
    'cooldown': 1,
    'min_lr': 1e-6
}

trainer = FRTrainerPlateau(
    param_config=default_param_config,
    net_config=default_net_config,
    dataset_config=default_dataset_config,
    optim_params_func=get_optim_params,
    scheduler_params=scheduler_params,
    eval_transform=eval_transform
)

trainer.train()
