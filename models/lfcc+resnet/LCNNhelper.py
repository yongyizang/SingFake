import torch.nn as nn
import torch


class MaxFeatureMap2D(nn.Module):
    """ Max feature map (along 2D)
    MaxFeatureMap2D(max_dim=1)
    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)
    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)
    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)
    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """

    def __init__(self):
        super(MaxFeatureMap2D, self).__init__()

    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)

        # shape = list(inputs.size())
        # # print(shape)
        # shape[1] = shape[1] // 2
        # # print(shape)
        # shape.insert(1, 2)
        # # print(shape)

        lst = torch.split(inputs, inputs.shape[1]//2, dim=1)
        # print(lst[0].shape)
        inputs = torch.stack(lst, dim=1)
        # print(inputs.shape)

        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.max(1)
        return m