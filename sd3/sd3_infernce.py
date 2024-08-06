from diffusers import StableDiffusion3Pipeline
import torch
from PIL import Image

from settings import PRELOAD_MODELS
from utils import device_checker


class SD3Model:
    def __init__(self, base_model_path: str, device: str =  "cuda:0"):
        self.base_model_path = base_model_path
        self.device = device_checker(device)
        self.base = None
        self.n_steps = 28
        self.guidance_scale = 7.0
        self.loaded = self._preload_models("SD3" in PRELOAD_MODELS)

    def _preload_models(self, preload: bool = False):

        if preload:
            self._load_base()
            print("All model loaded")
        return preload

    def load(self):
        self.loaded = self._preload_models(True)

    def unload(self):
        if self.base is not None:
            self.base.to('cpu')
            del self.base
        torch.cuda.empty_cache()
    def _load_base(self):
        if self.base is None:

            self.base = StableDiffusion3Pipeline.from_pretrained(
                self.base_model_path,
                text_encoder_3=None,
                tokenizer_3=None,
                torch_dtype=torch.float16
            )
            self.base = self.base.to(self.device)
            # Optionally compile the base unet
            # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
            print("Base model loaded.")
            return

    def run_sd3_text2img(self, prompt: str):
        if not self.loaded:
            self.loaded = self._preload_models(True)

        # Run both experts
        image = self.base(
            prompt=prompt,
            num_inference_steps=self.n_steps,
            guidance_scale=self.guidance_scale,
        ).images[0]

        # Convert the image to a format that can be saved to a buffer
        if isinstance(image, torch.Tensor):
            image = Image.fromarray((image * 255).cpu().numpy().astype('uint8'))

        return image
