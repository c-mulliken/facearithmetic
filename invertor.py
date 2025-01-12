import torch
import torch.nn as nn
from torchinfo import summary
from copy import deepcopy
import numpy as np
from PIL import Image
import sys
import io

sys.path.append('/home/coby/Repositories/facearithmetic')

import stylegan3.dnnlib as dnnlib
import stylegan3.legacy as legacy

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Invertor():
    def __init__(self, network_pkl='models/stylegan3-r-ffhqu-256x256.pkl'):
        with dnnlib.util.open_url(network_pkl) as f:
            self.D = legacy.load_network_pkl(f)['D']

model = Invertor().D.to(device)
Invertor.b4.out = nn.Linear(512, 512)

for param in Invertor.parameters():
  param.requires_grad = False
for param in Invertor.b4.conv.parameters():
  param.requires_grad = True


# data generator functions
def gen_data(num_samples, G=G):
    z_samps = torch.randn(num_samples, 512, device=device)
    x_samps = G(z_samps, None)
    return x_samps, z_samps