import logging
import math
import warnings


import torch
import torch.nn as nn
import torch.nn.functional as F

class LOA(nn.Module):
    def __init__(self, inplanes, pool='att', fusions=['channel_add', 'channel_mul']):
        super(LOA, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = inplanes // 4
        self.pool = pool
        self.fusions = fusions
        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
        )
        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
        )

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        input_x = input_x.view(batch, channel, height * width)
        input_x = input_x.unsqueeze(1)
        context_mask = self.conv_mask(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.unsqueeze(3)
        context = torch.matmul(input_x, context_mask)
        context = context.view(batch, channel, 1, 1)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
        out = x * channel_mul_term
        channel_add_term = self.channel_add_conv(context)
        out = out + channel_add_term
        return out



# if __name__ == "__main__":
#
#     num_feat = 32
#     batch_size = 8
#     image_size = 224
#
#     ImgLoader = torch.rand([batch_size, num_feat, image_size, image_size])
#
#     model = LOA(
#         inplanes=num_feat,
#     )
#     x = model(ImgLoader)
#
#     num_params = 0
#     for p in model.parameters():
#         if p.requires_grad:
#             num_params += p.numel()
#     print(f"Number of parameter {num_params / 10 ** 6:0.2f}")