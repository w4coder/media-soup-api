from diffusers import DiffusionPipeline
import torch
from PIL import Image

from settings import PRELOAD_MODELS
from utils import device_checker


class SDXLModel:
    def __init__(self, base_model_path: str, refiner_model_path: str, device: str = "cuda:0"):
        self.base_model_path = base_model_path
        self.refiner_model_path = refiner_model_path
        self.device = device_checker(device)
        self.base = None
        self.refiner = None
        self.n_steps = 40
        self.high_noise_frac = 0.8
        self.loaded = self._preload_models("SDXL" in PRELOAD_MODELS)

    def _preload_models(self, preload: bool = False):
        if preload:
            self._load_base()
            self._load_refiner()
            print("All model loaded")
        return preload

    def load(self):
        self.loaded = self._preload_models(True)

    def unload(self):
        if self.base is not None:
            self.base.to('cpu')
            del self.base
        if self.refiner is not None:
            self.refiner.to('cpu')
            del self.refiner
        torch.cuda.empty_cache()
        print("sd3 unloaded")

    def _load_base(self):
        if self.base is None:
            self.base = DiffusionPipeline.from_pretrained(
                self.base_model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
            )
            self.base.to(self.device)
            # Optionally compile the base unet
            # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
            print("sdxl model loaded.")

    def _load_refiner(self):
        if self.refiner is None:
            if self.base is None:
                self._load_base()
            self.refiner = DiffusionPipeline.from_pretrained(
                self.refiner_model_path,
                text_encoder_2=self.base.text_encoder_2,
                vae=self.base.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self.refiner.to(self.device)
            # Optionally compile the refiner unet
            # self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)
            print("Refiner model loaded.")

    def run_sdxl_text2img(self, prompt: str):
        if not self.loaded:
            self.loaded = self._preload_models(True)

        # Run both experts
        image = self.base(
            prompt=prompt,
            num_inference_steps=self.n_steps,
            denoising_end=self.high_noise_frac,
            output_type="latent",
        ).images
        image = self.refiner(
            prompt=prompt,
            num_inference_steps=self.n_steps,
            denoising_start=self.high_noise_frac,
            image=image,
        ).images[0]

        # Convert the image to a format that can be saved to a buffer
        if isinstance(image, torch.Tensor):
            image = Image.fromarray((image * 255).cpu().numpy().astype('uint8'))

        return image
