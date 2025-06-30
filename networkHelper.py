# 导入各种包
import torch
from torch import nn
import numpy as np
import math
from inspect import isfunction
from einops.layers.torch import Rearrange
from torchvision.transforms import Compose, Lambda, ToPILImage


def exists(x):
    """
    判断数值是否为空
    :param x: 输入数据
    :return: 非空True，反之False
    """
    return x is not None

def default(val, d):
    """
    该函数的目的是提供一个简单的机制来获取给定变量的默认值
    如果val存在，则返回该值。如果不存在，则使用d函数提供默认值，
    或者如果d不是一个函数，则返回d
    :param val: 需要判断的变量
    :param d: 提供默认值的变量或函数
    :return:
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d

def num_to_groups(num, divisor):
    """
    该函数的目的是将数字分成若干组，每组大小都为divisior，并返回一个列表，
    其中包含所有这些组的大小。如果num不能完全被divisor整除，则最后一组大小将小于divisor
    :param num:
    :param divisor:
    :return:
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class Residual(nn.Module):
    def __init__(self, fn):
        """
        残差连接模块
        :param fn:激活函数类型
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """
        残差连接前馈
        :param x:输入数据
        :param args:
        :param kwargs:
        :return:f(x) + x
        """
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out=None):
    """
    这个上采样模块的作用是将输入张量的尺寸在宽高上放大2倍
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),            #先使用最近邻填充将数据在长宽上翻倍
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),    #再使用卷积对翻倍后的数据提取局部相关关系填充
    )

def Downsample(dim, dim_out=None):
    """
    下采样模块的作用是将输入张量的分辨率降低，通常用于在深度学习模型中对特征图进行下采样
    在这个实现中，下采样操作的方式是使用一个2x2的最大池化操作，将输入张量的宽和高都缩小一半，
    然后再使用上述的变换和卷积操作得到输出张量。由于这个实现使用了形状变换操作，因此没有使用传统的卷积或池化操作进行下采样，
    从而避免了在下采样过程中丢失信息的问题
    :param dim:
    :param dim_out:
    :return:
    """
    return nn.Sequential(
        #将输入张量的形状由（batch_size, channel, height, width）变换为（batch_size, channel * 4, height/2, width/2）
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        # 对变换后的张量进行的卷积操作，将通道数从 dim * 4（即变换后的通道数）降到 dim（即指定的输出通道数），得到输出张量。
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )

# 位置嵌入 - Position embeddings http://www.egbenz.com/#/my_article/19
# 因为每次计算噪声的时候都需要传入时刻t，那每次应该都得不一样才行
# 下面代码就是transformer中的正余弦编码
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def extract(a, t, x_shape):
    """
    从给定的张量a中检索特定的元素.t是一个包含要检索的索引张量，
    这些索引对应于a张量中的元素。这个函数输出是一个张量，
    包含了t张量中每个索引对应的a张量中的元素
    :param a:
    :param t:
    :param x_shape:
    :return:
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    A = len(x_shape)-1
    return out.reshape(batch_size, *((1,) * (len(x_shape)-1))).to(t.device)


# show a random one
reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])


