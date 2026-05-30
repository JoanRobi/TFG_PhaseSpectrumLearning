from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.feature import local_binary_pattern
from torchvision.transforms.functional import pil_to_tensor, rgb_to_grayscale


def obre_imatgeRGB(ruta_imatge, mida=None):

    img = Image.open(ruta_imatge).convert("RGB")
    
    img_rgb = pil_to_tensor(img).float() / 255.0
    return img_rgb


def calcula_MagnitudFase(img_rgb):
    img_gris = rgb_to_grayscale(img_rgb, num_output_channels=1)
    # dim = -2,-1 perque volem que faci la transformacio adamunt les dues darreres dimensions del tensor
    fft2 = torch.fft.fft2(img_gris, dim=(-2, -1))
    # empram fftshift per reorganitzar les freqüencies de tal manera que les mes baixes queden al centre
    fft2_reorg = torch.fft.fftshift(fft2, dim=(-2, -1))
    # la magnitud ens diu la presencia de cada valor de la im atge (la força) 
    # magnitud = quant hi ha de cada freqüencia
    magnitud = torch.abs(fft2_reorg)
    magnitud_log = torch.log1p(magnitud)
    # las fase va mes relacionada a com estan alineades les estrcutures
    # fase = organitzacio d'aquestes freqüencies
    fase = torch.angle(fft2_reorg)
    fase_norm = (fase + torch.pi) / (2 * torch.pi)

    return img_gris, magnitud_log, fase_norm


def calcula_lbp(img_gris, punts=8, radi=1, metode="uniform"):

    img_gris_np = img_gris.squeeze(0).cpu().numpy()
    img_gris_u8 = (img_gris_np * 255).astype(np.uint8)

    lbp = local_binary_pattern(img_gris_u8, P=punts, R=radi, method=metode)
    lbp_tensor = torch.from_numpy(lbp).float().unsqueeze(0)

    lbp_tensor = (lbp_tensor - lbp_tensor.min()) / (
        lbp_tensor.max() - lbp_tensor.min() + 1e-8
    )

    return lbp_tensor


def construeix_x0(img_rgb):
    """
        Nota: construim la representacio augmentada que servira com a entrada del model
    """
    img_gris, magnitud_log, fase_norm = calcula_MagnitudFase(img_rgb)
    lbp_tensor = calcula_lbp(img_gris)
    # concatenam tensors
    x0 = torch.cat([img_rgb, magnitud_log, lbp_tensor], dim=0)

    return x0, magnitud_log, fase_norm, lbp_tensor