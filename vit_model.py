# ---- Nested function ---- #
import os
import pandas as pd
import wfdb
import ast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from pprint import pprint
from collections import Counter
import math
from copy import deepcopy
import random

# # ---- BWR ---- #
# import bwr
# import emd
# import pywt
# ---- Scipy ---- #
from scipy import signal
from scipy.signal import butter, lfilter, freqz, filtfilt
from scipy.fftpack import fft
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


# ---- PyTorch ---- #
import torch
import torchvision
from torch import nn
from torch import optim
from torch import functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision.transforms import ToTensor
from torch.nn.functional import softmax
from torch.nn.parallel import DistributedDataParallel
from pytorchtools import EarlyStopping
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torchvision.ops as ops
import tensorboard
from tensorboardX import SummaryWriter

# ---- Scikit Learn ---- #
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import KFold


# ---- Matplotlib ---- #
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Summary ---- #
import pytorch_model_summary

class ViTEmbeddings(nn.Module):
    def __init__(self, in_channel, emb_size, patch_size, dropout=0.0):
        super().__init__()
        
        self.patch_size= patch_size
        self.emb_size= emb_size
        
        self.patch_embeddings = nn.Sequential(
            nn.Conv1d(in_channel, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange("b e (p) -> b (p) e")
        )
        self.dropout = nn.Dropout(dropout)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_size))
        num_patches = 5000//patch_size
        self.positions = nn.Parameter(torch.randn(1, num_patches +1, self.emb_size)) ## Num Patches....어케하지
    
    def forward(self, x):
        input_shape = x.shape # B C L
        embeddings = self.patch_embeddings(x)
        cls_token = repeat(self.cls_token, "() n e -> b n e", b=input_shape[0])
        x = torch.cat([cls_token, embeddings], dim=1)

        x += self.positions
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, emb_size: int = 768, expansion: int = 4, dropout=0.0, mlp_dim=256):
        super().__init__()
        self.mlps = nn.Sequential(
            nn.Linear(emb_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = self.mlps(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, emb_size: int=768, num_heads: int=6,f_expansion: int=4, f_dropout=0.0, dropout=0.0, sd_survive=0.0,**kwargs):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout, bias=False, batch_first=True)
        self.lnorm_b = nn.LayerNorm(emb_size)
        self.lnorm_a = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(0.2)
        self.mlp = MLP(emb_size, expansion=f_expansion, dropout=f_dropout, mlp_dim=256)
        self.sd_survive=sd_survive
        self.actv1 = nn.GELU()
        self.actv2 = nn.GELU()
    
        self.stochasticLayer = ops.StochasticDepth(sd_survive, "row")
        
    def forward(self, x):
        x_norm = self.lnorm_b(x)
        x_norm, _ = self.attn(x_norm, x_norm, x_norm)
        # x_norm = self.dropout(x_norm)
        x_norm = self.stochasticLayer(x_norm)
        x = torch.add(x_norm, x)
        x2_norm = self.lnorm_a(x)
        x2_norm = self.mlp(x2_norm)
        x2_norm = self.stochasticLayer(x2_norm)
        # x2_norm = self.dropout(x2_norm)
        x2 = torch.add(x2_norm, x)
        return x2
    
    def get_attention_scores(self, inputs):
        x = self.lnorm_b(inputs)
        output, weight = self.attn(x, x, x, average_attn_weights=False)
        print(output.shape, weight.shape)
        return weight

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int=768, n_classes: int=5):
        super().__init__(
            # Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(emb_size),
            # nn.GELU(),
            nn.Linear(emb_size, n_classes)
            # nn.Dropout(0.2)
        )

class ViT(nn.Module):
    def __init__(self, in_channel: int= 12, patch_size: int= 20, emb_size: int= 768, num_heads: int= 6, n_classes: int= 5, depth: int= 6, mlp_dim: int=256):
        super().__init__()
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.n_classes = n_classes
        self.depth = depth
        self.mlp_dim = 256
        self.sd_survive = np.linspace(0, 0.9, depth)
        self.Blocks = nn.ModuleList([
            EncoderBlock(emb_size=self.emb_size, mlp_dim=self.mlp_dim, sd_survive=self.sd_survive[i]) for i in range(depth)
        ])
        self.Embeddings = ViTEmbeddings(in_channel, emb_size, patch_size)
        self.ClassificationHead = ClassificationHead(emb_size, n_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.Embeddings(x)
        for block in self.Blocks:
            x = block(x)
        x = x[:,0]
        result = self.ClassificationHead(x)
        result = nn.Sigmoid(result)
        return result
    
    def get_last_selfattention(self, inputs):
        x = self.Embeddings(inputs)
        for block in self.Blocks[:-1]:
            x = block(x)
        return self.Blocks[-1].get_attention_scores(x)