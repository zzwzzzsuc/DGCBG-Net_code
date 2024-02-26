import torch
import torch.nn as nn

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, SR, GT):
        num = GT.size(0)
        smooth = 1

        m1 = SR
        m2 = GT
        intersection = (m1 * m2)

        loss = 1 - (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        loss = loss.sum() / num


        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        return loss.mean()

class BCELoss(torch.nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, SR, GT):
        criterion = nn.BCELoss()
        loss = criterion(SR, GT)

        return loss

class Dice_and_BCELoss(nn.Module):
    def __init__(self):
        super(Dice_and_BCELoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = BCELoss()

    def forward(self, SR, GT):
        loss = 0.8 * self.dice_loss(SR, GT) + 0.2 * self.focal_loss(SR, GT)
        # loss = self.dice_loss(input, target)
        return loss



class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, SR, GT):
        loss = self.dice_loss(SR, GT) + self.focal_loss(SR, GT)
        # loss = self.dice_loss(input, target)
        return loss



