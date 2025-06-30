import nntplib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from basicsr.networkHelper import *
import torch
import copy
import basicsr.models.Upsamplers as Upsamplers
import torch.nn.functional as F
from basicsr.BCHW2BLC import bchw_to_blc, blc_to_bchw
import matplotlib
matplotlib.use('TkAgg')


class CCB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CCB, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels

        self.shared_cat = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch // 2, 1),
            nn.BatchNorm2d(self.out_ch // 2),
            nn.GELU()
        )
        self.shared_cascade = PDW(in_channel=self.in_ch, num_feat=self.out_ch // 2, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)


        self.Conv1 = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch // 2, 1),
            nn.BatchNorm2d(self.out_ch // 2),
            nn.GELU()
        )


        self.fusion  = nn.Sequential(
            nn.Conv2d(self.out_ch * 2, self.out_ch, 1),
            nn.BatchNorm2d(self.out_ch),
            nn.GELU()
        )

        self.USCAB = USCAB(in_channel=self.in_ch, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

    def forward(self, input):
        h = input
        x_cat_1 = self.shared_cat(h)
        x_cascade_1 = self.shared_cascade(h)
        x1_level = torch.cat([x_cat_1, x_cascade_1], dim=1)

        x_cat_2 = self.shared_cat(x1_level)
        x_cascade_2 = self.shared_cascade(x1_level)
        x2_level = torch.cat([x_cat_2, x_cascade_2], dim=1)

        x_cat_3 = self.shared_cat(x2_level)
        x_cascade_3 = self.shared_cascade(x2_level)
        x3_level = torch.cat([x_cat_3, x_cascade_3], dim=1)

        x4 = self.Conv1(x3_level)

        X_Fusion = torch.cat([x_cat_1, x_cat_2, x_cat_3, x4], dim=1)
        X = self.fusion(X_Fusion)
        uscab = self.USCAB(X)
        out = uscab + input
        return out


def STD(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def concarenate_feature(x, num_block):
    if num_block == 4:
        y = torch.cat([x[0], x[1], x[2], x[3]], dim=1)
    elif num_block == 8:
        y = torch.cat([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]], dim=1)
    elif num_block == 12:
        y = torch.cat([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]], dim=1)
    return y


def concarenate_piece(x):
    return torch.cat([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8],
                      x[9], x[10], x[11], x[12], x[13], x[14], x[15]], dim=1)


def concarenate_piece(x):
    return torch.cat([x[0], x[1], x[2], x[3]], dim=1)

def swish(x):
    return x * torch.sigmoid(x)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        b, c, h, w = x.size()
        r = self.downscale_factor
        assert h % r == 0 and w % r == 0, "Height and Width must be divisible by downscale factor"
        out_c = c * (r ** 2)
        out_h = h // r
        out_w = w // r
        x = x.view(b, c, out_h, r, out_w, r)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, out_c, out_h, out_w)
        return x


class USCAB(nn.Module):
    def __init__(self, in_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode="zeros"):
        super(USCAB, self).__init__()

        self.down_block = nn.Sequential(
              # 1st downsample: (C, H, W) -> (4C, H/2, W/2)
            nn.Conv2d(in_channel, in_channel // 4, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode),
            nn.GELU(),
            PixelUnshuffle(2)
        )

        self.up_block = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * 4, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode),
            nn.PixelShuffle(2),
            nn.GELU()
        )

        self.contrast = STD

        self.avg_pool = nn.AdaptiveAvgPool2d(1)


    def forward(self, input):
        x = self.down_block(input)
        x = self.down_block(x)
        x = self.down_block(x)

        std = self.contrast(x)

        x = self.up_block(x)
        x = self.up_block(x)
        x = self.up_block(x)
        y2 = x + input
        W  = F.normalize(y2, p=2, dim=1, eps=1e-6)
        Fe = W * input
        AVG = self.avg_pool(Fe)
        Fuscab = (AVG + std) * Fe
        return Fuscab



class PDW(nn.Module):
    def __init__(self, in_channel, num_feat, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode="zeros"):
        super(PDW, self).__init__()

        # PW
        self.pw = nn.Conv2d(in_channels=in_channel, out_channels=num_feat, kernel_size=(1, 1), stride=1, padding=0, dilation=dilation, groups=1, bias=False)
        # DW
        self.dw = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=num_feat, bias=bias, padding_mode=padding_mode)
        self.act = nn.GELU()


    def forward(self, x):
        x = self.pw(x)
        x = self.act(x)
        x = self.dw(x)
        x = self.act(x)
        return x

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Mlp(nn.Module):
    def __init__(self, num_feat, hidden_size , dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(num_feat, int(hidden_size / 8))
        self.fc2 = nn.Linear(int(hidden_size / 8), num_feat)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, num_feat, hidden_size, attention_dropout_rate):
        super(Attention, self).__init__()
        self.num_attention_heads = int(hidden_size / num_feat)
        self.attention_head_size = num_feat
        self.hidden_size = hidden_size
        self.qkv = nn.Linear(num_feat, hidden_size*3)
        self.query = nn.Linear(num_feat, hidden_size)
        self.key = nn.Linear(num_feat, hidden_size)
        self.value = nn.Linear(num_feat, hidden_size)

        self.out = nn.Linear(hidden_size, num_feat)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.GELU()



    def forward(self, hidden_states):

        B, N, C = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(B, N, 3, self.num_attention_heads, self.hidden_size * 3 // (3 * self.num_attention_heads)).permute(2, 0, 3, 1, 4)
        query_layer, key_layer, value_layer = qkv[0], qkv[1], qkv[2]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.act(attention_output)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Block(nn.Module):
    def __init__(self, hidden_size,  num_feat, dropout_rate, attention_dropout_rate):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(num_feat, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(num_feat, eps=1e-6)
        self.ffn = Mlp(num_feat=num_feat, hidden_size=hidden_size, dropout_rate=dropout_rate)
        self.attn = Attention(num_feat=num_feat, hidden_size=hidden_size, attention_dropout_rate=attention_dropout_rate)
        self.proj = nn.Linear(in_features=num_feat, out_features=num_feat, bias=True)
        self.act = nn.GELU()


    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h
        z = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + z
        return x


class Encoder(nn.Module):
    def __init__(self, num_block, num_feat, hidden_size, dropout_rate,  attention_dropout_rate):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(num_feat, eps=1e-6)
        for _ in range(num_block):
            layer = Block(hidden_size=hidden_size, num_feat=num_feat,  dropout_rate=dropout_rate, attention_dropout_rate=attention_dropout_rate)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        #print(hidden_states.shape)
        #Feature_cat = []
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
            encoded = self.encoder_norm(hidden_states)
            # y = blc_to_bchw(encoded, x_size)
            # Feature_cat.append(y)
        return encoded

class Transformer(nn.Module):
    def __init__(self, num_block, num_feat, hidden_size, drop_rate, attention_dropout_rate):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_block=num_block, num_feat=num_feat, hidden_size=hidden_size,  dropout_rate=drop_rate, attention_dropout_rate=attention_dropout_rate)

    def forward(self, inputs_ids):
        encoded = self.encoder(inputs_ids)
        return encoded



class RCTB(nn.Module):
    def __init__(self,  num_feat, hidden_size, drop_rate, attention_drop_rate):
        super(RCTB, self).__init__()
        self.norm_first = nn.LayerNorm(num_feat)
        self.drop_first = nn.Dropout(p=drop_rate)
        self.transformer = nn.ModuleList()
        self.proj = nn.Linear(in_features=num_feat, out_features=num_feat, bias=True)
        self.act = nn.GELU()
        for _ in range(4):
            rctb = Transformer(num_block=int(2), num_feat=num_feat, hidden_size=hidden_size, drop_rate=drop_rate, attention_dropout_rate=attention_drop_rate)
            self.transformer.append(copy.deepcopy(rctb))

    def forward(self, input):
        # 首先按照行分割
        Row = input.split(input.shape[2] // 4, dim=2)
        piece_attention_list_ROW = []
        for i, (VIT, row_feat) in enumerate(zip(self.transformer, Row)):
            r_x = bchw_to_blc(row_feat)
            r_x = self.norm_first(r_x)
            r_x = self.drop_first(r_x)
            r_x = VIT(r_x)
            piece_attention_list_ROW.append(r_x)
        r_x = concarenate_piece(piece_attention_list_ROW)
        r_x = self.proj(r_x)
        r = self.act(r_x)
        r = blc_to_bchw(r, (input.shape[2], input.shape[3]))
        Col = r.split(r.shape[3] // 4, dim=3)
        piece_attention_list_COL = []
        for j, (VIT, col_feat) in enumerate(zip(self.transformer, Col)):
            c_x = bchw_to_blc(col_feat)
            c_x = self.norm_first(c_x)
            c_x = self.drop_first(c_x)
            c_x = VIT(c_x)
            piece_attention_list_COL.append(c_x)
        c_x = concarenate_piece(piece_attention_list_COL)
        c_x = self.proj(c_x)
        c = self.act(c_x)
        Output = blc_to_bchw(c, (input.shape[2], input.shape[3]))
        return Output



class EIRSR(nn.Module):
    def __init__(self,
            upscale=2,
            num_block=4,
            num_feat=16,
            num_in_ch=1,
            num_out_ch=1,
            drop_rate=0.1,
            hidden_size=64,
            attention_drop_rate=0.0
    ):
        super().__init__()
        self.conv_pdw_1 = PDW(num_in_ch, num_feat, 3, 1, 1)
        self.C1 = CCB(in_channels=num_feat, out_channels=num_feat)
        self.C2 = CCB(in_channels=num_feat, out_channels=num_feat)
        self.C3 = CCB(in_channels=num_feat, out_channels=num_feat)
        self.C4 = CCB(in_channels=num_feat, out_channels=num_feat)
        self.C5 = CCB(in_channels=num_feat, out_channels=num_feat)
        self.C6 = CCB(in_channels=num_feat, out_channels=num_feat)
        self.E1 = RCTB(num_feat=num_feat, hidden_size=hidden_size, drop_rate=drop_rate, attention_drop_rate=attention_drop_rate)
        self.E2 = RCTB(num_feat=num_feat, hidden_size=hidden_size, drop_rate=drop_rate, attention_drop_rate=attention_drop_rate)
        self.E3 = RCTB(num_feat=num_feat, hidden_size=hidden_size, drop_rate=drop_rate, attention_drop_rate=attention_drop_rate)
        self.E4 = RCTB(num_feat=num_feat, hidden_size=hidden_size, drop_rate=drop_rate, attention_drop_rate=attention_drop_rate)
        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 3, 1, 1)
        self.c2 = self.c2 = PDW(num_feat, num_feat, 3, 1, 1)
        self.GELU = nn.GELU()
        self.pool = nn.MaxPool2d(2, 2)
        self.num_out_ch = num_out_ch
        self.UP = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
        self.Conv_CAT = nn.Conv2d(num_feat, num_feat, 1, 1)
        self.last_conv = PDW(num_feat, num_feat, 3, 1, 1)

    def forward(self, input):
        feat1 = self.conv_pdw_1(input)
        out_E1 = self.C1(feat1)
        out_E2 = self.C2(out_E1)
        out_E3 = self.C3(out_E2)
        out_E4 = self.C4(out_E3)
        out_E5 = self.C5(out_E4)
        out_E6 = self.C6(out_E5)
        out_cat = torch.cat([out_E1, out_E2, out_E3, out_E4, out_E5, out_E6], dim=1)
        out_E = self.c1(out_cat)
        out_E = self.GELU(out_E)
        Before_Atten = self.c2(out_E) + feat1
        E1 = self.E1(Before_Atten)
        E2 = self.E2(E1)
        E3 = self.E3(E2)
        E4 = self.E4(E3)
        x = self.Conv_CAT(E4)
        x = self.GELU(x)
        x = self.last_conv(x) + Before_Atten
        output = self.UP(x)
        return output








