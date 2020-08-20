import torch
import torch.nn as nn
import torchvision
from utils.general import UnNormalize


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class PerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, unnorm=True):
        super(PerceptualLoss, self).__init__()
        vgg_model = torchvision.models.vgg16(pretrained=True)
        vgg_pretrained_features = vgg_model.eval().features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        del vgg_model

        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.resize = resize
        self.unnorm = unnorm
        self.unnormalize = UnNormalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

    def forward(self, input, target):

        if self.unnorm:
            input = self.unnormalize(input)
            target = self.unnormalize(target)

        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std

        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        h_inp = self.slice1(input)
        h_tar = self.slice1(target)
        h_relu1_2 = h_inp, h_tar

        h_inp = self.slice2(h_inp)
        h_tar = self.slice2(h_tar)
        h_relu2_2 = h_inp, h_tar

        h_inp = self.slice3(h_inp)
        h_tar = self.slice3(h_tar)
        h_relu3_3 = h_inp, h_tar

        h_inp = self.slice4(h_inp)
        h_tar = self.slice4(h_tar)
        h_relu4_3 = h_inp, h_tar

        loss = 0.
        input_dim = input.shape[0]
        for out in [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]:
            loss += torch.nn.functional.l1_loss(out[0], out[1])

        return loss
