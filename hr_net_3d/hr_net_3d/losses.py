import torch
import torch.nn as nn


class CustomCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(CustomCrossEntropyLoss, self).__init__()

        self.loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, logits, labels):
        # prepare labels
        bg = (1 - torch.sum(labels, 1, keepdim=True))
        labels = torch.cat([bg, labels], 1)

        loss = self.loss(logits, labels)

        return loss


class CustomMSELoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(CustomMSELoss, self).__init__()

        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, logits, labels):
        # prepare logits
        logits_sm = torch.softmax(logits, 1)

        # prepare labels
        bg = (1 - torch.sum(labels, 1, keepdim=True))
        labels = torch.cat([bg, labels], 1)

        loss = self.loss(logits_sm, labels)

        return loss
