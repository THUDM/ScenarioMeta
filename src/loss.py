import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.m = nn.LogSigmoid()

    def forward(self, positives, negatives):
        return -self.m(positives - negatives).mean()


class MyHingeLoss(nn.Module):
    def __init__(self, margin=0.0):
        nn.Module.__init__(self)
        self.m = nn.MarginRankingLoss(margin=margin)

    def forward(self, positives, negatives):
        labels = positives.new_ones(positives.size())
        return self.m(positives, negatives, labels)


class MyBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.m = nn.BCEWithLogitsLoss()

    def forward(self, positives, negatives):
        values = torch.cat((positives, negatives), dim=-1)
        labels = torch.cat((positives.new_ones(positives.size()),
                            negatives.new_zeros(negatives.size())), dim=-1)
        return self.m(values, labels)
