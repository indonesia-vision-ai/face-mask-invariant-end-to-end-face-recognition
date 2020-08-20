import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import pi, cos, sin


class LiArcFace(nn.Module):

    def __init__(self, embedding_size, classnum, m=0.45, s=64.0):
        super(LiArcFace, self).__init__()
        self.weight = nn.Parameter(torch.empty(classnum, embedding_size))
        nn.init.xavier_normal_(self.weight)
        self.m = m
        self.s = s
        self.pi = torch.tensor(pi)

    def forward(self, input, label):
        W = F.normalize(self.weight)
        input = F.normalize(input)
        cosine = input @ W.t()
        theta = torch.acos(cosine)
        m = torch.zeros_like(theta)
        m.scatter_(1, label.view(-1, 1), self.m)
        logits = self.s * (self.pi - 2 * (theta + m)) / self.pi

        return logits


class Softmax(nn.Module):

    def __init__(self, embedding_size, classnum):
        super(Softmax, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(embedding_size, classnum))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)


    def l2_norm(self, input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)

        return output


    def forward(self, input, label):
        weight_norm = self.l2_norm(self.weight, axis=0)
        cos_theta = torch.mm(input, weight_norm)

        return cos_theta


class ArcFace(nn.Module):

    def __init__(self, embedding_size, classnum, s=64., m=0.5):
        super(ArcFace, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(embedding_size, classnum))

        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m
        self.threshold = math.cos(pi - m)


    def l2_norm(self, input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)

        return output


    def forward(self, input, label):
        nB = len(input)
        weight_norm = self.l2_norm(self.weight, axis=0)

        cos_theta = torch.mm(input, weight_norm)

        cos_theta = cos_theta.clamp(-1, 1)
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)

        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s

        return output



class ArcFaceAlt(nn.Module):

    def __init__(self, in_features, out_features, s=64.0, m=0.50,
                 easy_margin=False):
        super(ArcFaceAlt, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------- cos(theta) & phi(theta) -----------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # -------------------- convert label to one-hot -----------------------
        one_hot = torch.zeros(cosine.size()).to(cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # NOTE you can use torch.where if your torch.__version__ is 0.4
        # torch.where(out_i = {x_i if condition_i else y_i)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        output *= self.s

        return output
