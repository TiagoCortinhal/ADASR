from torch import nn
import torch


class TVLoss(nn.Module):
    """
    Total variation loss.
    """

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, yhat, y):
        bsize, chan, height, width = y.size()
        dyh = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
        dyhath = torch.abs(yhat[:, :, 1:, :] - yhat[:, :, :-1, :])

        dyw = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
        dyhatw = torch.abs(yhat[:, :, :, 1:] - yhat[:, :, :, :-1])
        error = (torch.norm(dyh - dyhath, 1) / height) + (torch.norm(dyw - dyhatw, 1) / width)

        return error