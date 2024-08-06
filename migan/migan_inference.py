import os
import warnings
from glob import glob
from pathlib import Path
from fastapi import  HTTPException
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from migan.lib.model_zoo.migan_inference import Generator as MIGAN
from migan.lib.model_zoo.comodgan import (
    Generator as CoModGANGenerator,
    Mapping as CoModGANMapping,
    Encoder as CoModGANEncoder,
    Synthesis as CoModGANSynthesis
)
from utils import device_checker

warnings.filterwarnings("ignore")


class MIGAN_OR:
    # Define allowed models
    _ALLOWED_MODELS = {
        'migan-256-places2': 'models/migan/migan_256_places2.pt',
        # Replace with actual paths or model loading functions
        'migan-512-places2': 'models/migan/migan_512_places2.pt',
        'migan-256-ffhq': 'models/migan/migan_256_ffhq.pt',
        'comodgan-256-places2': 'models/migan/comodgan_256_places2.pt',
        # Replace with actual paths or model loading functions
        'comodgan-512-places2': 'models/migan/comodgan_512_places2.pt',
        'comodgan-256-ffhq': 'models/migan/comodgan_256_ffhq.pt',
        'powerpaintv2':'powerpaintv2'
    }

    def __init__(self,  device: str = "cuda:0"):
        self.model_name = None
        self.model_path = None
        self.device = device_checker(device)
        self.model = None

    def load(self):
        print("migan loaded")

    def unload(self):
        print("migan unloaded")

    def _load_model(self,model_name: str, model_path: str):
        try:
            if "migan-256" in model_name:
                resolution = 256
                model = MIGAN(resolution=256)
            elif "migan-512" in model_name:
                resolution = 512
                model = MIGAN(resolution=512)
            elif "comodgan-256" in model_name:
                resolution = 256
                comodgan_mapping = CoModGANMapping(num_ws=14)
                comodgan_encoder = CoModGANEncoder(resolution=resolution)
                comodgan_synthesis = CoModGANSynthesis(resolution=resolution)
                model = CoModGANGenerator(comodgan_mapping, comodgan_encoder, comodgan_synthesis)
            elif "comodgan-512" in model_name:
                resolution = 512
                comodgan_mapping = CoModGANMapping(num_ws=16)
                comodgan_encoder = CoModGANEncoder(resolution=resolution)
                comodgan_synthesis = CoModGANSynthesis(resolution=resolution)
                model = CoModGANGenerator(comodgan_mapping, comodgan_encoder, comodgan_synthesis)
            else:
                raise Exception("Unsupported model name.")

            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            self.model_name = model_name
            self.model_path = model_path
            return model
        except Exception as e:
            raise Exception(e)

    @staticmethod
    def read_mask(mask: Image.Image, invert: bool = True) -> Image.Image:

        mask = MIGAN_OR.resize(mask, max_size=512, interpolation=Image.NEAREST)
        mask = np.array(mask)
        if len(mask.shape) == 3:
            if mask.shape[2] == 4:
                _r, _g, _b, _a = np.rollaxis(mask, axis=-1)
                mask = np.dstack([_a, _a, _a])
            elif mask.shape[2] == 2:
                _l, _a = np.rollaxis(mask, axis=-1)
                mask = np.dstack([_a, _a, _a])
            elif mask.shape[2] == 3:
                _r, _g, _b = np.rollaxis(mask, axis=-1)
                mask = np.dstack([_r, _r, _r])
        else:
            mask = np.dstack([mask, mask, mask])
        if invert:
            mask = 255 - mask
        mask[mask < 255] = 0
        new_mask = Image.fromarray(mask)

        # Display the image using matplotlib
        # plt.imshow(new_mask)
        # plt.axis('off')  # Turn off axis labels
        # plt.show()
        return new_mask

    @staticmethod
    def resize(image: Image.Image, max_size: int, interpolation=Image.BICUBIC) -> Image.Image:
        w, h = image.size
        if w > max_size or h > max_size:
            resize_ratio = max_size / w if w > h else max_size / h
            image = image.resize((int(w * resize_ratio), int(h * resize_ratio)), interpolation)
        return image

    @staticmethod
    def preprocess(img: Image.Image, mask: Image.Image, resolution: int) -> torch.Tensor:
        img = img.resize((resolution, resolution), Image.BICUBIC)
        mask = mask.resize((resolution, resolution), Image.NEAREST)
        img = np.array(img)
        mask = np.array(mask)[:, :, np.newaxis] // 255
        img = torch.Tensor(img).float() * 2 / 255 - 1
        mask = torch.Tensor(mask).float()
        img = img.permute(2, 0, 1).unsqueeze(0)
        mask = mask.permute(2, 0, 1).unsqueeze(0)
        x = torch.cat([mask - 0.5, img * mask], dim=1)
        return x

    def process_image(self, img: Image.Image, mask: Image.Image,model_name: str, model_path: str) -> Image.Image:

        if not self.model or (self.model_name is not model_name) or (self.model_path is not model_path):
            self.model = self._load_model(model_name=model_name,model_path=model_path)
        resolution = 512 if "512" in self.model_name else 256
        img_resized = self.resize(img, max_size=resolution)
        mask_resized = MIGAN_OR.read_mask(mask)
        # mask_resized = self.resize(mak_format, max_size=resolution, interpolation=Image.NEAREST)

        x = self.preprocess(img_resized, mask_resized.convert("L"), resolution)
        if "cuda" in self.device:
            x = x.to(self.device)
        with torch.no_grad():
            result_image = self.model(x)[0]
        result_image = (result_image * 0.5 + 0.5).clamp(0, 1) * 255
        result_image = result_image.to(torch.uint8).permute(1, 2, 0).detach().to("cpu").numpy()

        result_image = cv2.resize(result_image, dsize=img_resized.size, interpolation=cv2.INTER_CUBIC)
        shape = np.array(mask_resized).shape
        mask_resized = np.array(mask_resized) // 255
        composed_img = np.array(img_resized) * mask_resized + result_image * (1 - mask_resized)
        composed_img = Image.fromarray(composed_img.astype(np.uint8))

        return composed_img

    @property
    def ALLOWED_MODELS(self):
        return self._ALLOWED_MODELS
