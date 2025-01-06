import torch
import numpy as np
from PIL import Image
import sys
import io

sys.path.append('/home/coby/Repositories/facearithmetic')

import stylegan3.dnnlib as dnnlib
import stylegan3.legacy as legacy

# network_pkl = 'models/stylegan3-r-ffhqu-256x256.pkl'  # or wherever you placed it
# print(f'Loading networks from "{network_pkl}"...')
# device = torch.device('cuda')  # 'cuda' or 'cpu'
# with dnnlib.util.open_url(network_pkl) as f:
#     G = legacy.load_network_pkl(f)['G_ema'].to(device)
# with dnnlib.util.open_url(network_pkl) as f:
#     D = legacy.load_network_pkl(f)['D'].to(device)

class StyleGan():
    def __init__(self, network_pkl='models/stylegan3-r-ffhqu-256x256.pkl',
                gen=True, device=torch.device('cuda')):
        if gen:
            with dnnlib.util.open_url(network_pkl) as f:
                self.G = legacy.load_network_pkl(f)['G_ema'].to(device)
        else:
            with dnnlib.util.open_url(network_pkl) as f:
                self.D = legacy.load_network_pkl(f)['D'].to(device)

    def gen_img(self, z, truncation_psi):
        img = self.G(z, None, truncation_psi=truncation_psi)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].permute(1, 2, 0).cpu().numpy()
        return img
    
    def gen_pil(self, z, truncation_psi):
        img = self.G(z, None, truncation_psi=truncation_psi)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img)
        img = img.resize((128, 128))
        with io.BytesIO() as output:
          img.save(output, format="PNG")
          return output.getvalue()

    def generate_imgs(self, zs, truncation_psi):
        imgs = []
        for z in zs:
          imgs.append(self.gen_img(z, truncation_psi))
        return imgs

def output_to_img(output):
  img = (output * 127.5 + 128).clamp(0, 255).to(torch.uint8)
  img = img[0].permute(1, 2, 0).cpu().numpy()
  return img



def interpolate(z1, z2, num_steps):
  steps = []
  for i in range(num_steps):
    t = i / (num_steps - 1)
    z = (z1 * (1 - t)) + (z2 * t)
    steps.append(z)
  return steps

