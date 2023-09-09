import torch.nn as nn


class IoULoss(nn.Module):
    """
    该损失函数用于小论文中
    """
    def __init__(self, smooth=0.1):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()   # 计算交集

        total = (inputs + targets).sum()          # 计算并集
        union = total - intersection

        iouloss = 1 - (intersection + self.smooth) / (union + self.smooth)    # 计算IOULoss

        return iouloss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()  # 计算并集
        union = total - intersection

        dice_loss = 1 - ((2.0 * intersection + self.smooth) / (union + self.smooth))

        return dice_loss


class CombineIouAndDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(CombineIouAndDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()  # 计算交集

        total = (inputs + targets).sum()  # 计算并集
        union = total - intersection

        iouloss = 1 - (intersection + self.smooth) / (union + self.smooth)  # 计算IOULoss
        dice_loss = 1 - ((2.0 * intersection + self.smooth) / (union + self.smooth))
        combineloss = 0.5 * iouloss + 0.5 + dice_loss

        return combineloss