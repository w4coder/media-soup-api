from diffusers import FluxPipeline
import torch
from PIL import Image
from accelerate import PartialState
from settings import PRELOAD_MODELS
from utils import device_checker


class FluxSchnellModel:
    def __init__(self, base_model_path: str, device: str =  "cuda:0"):
        self.base_model_path = base_model_path
        self.device = device_checker(device)
        self.base = None
        self.n_steps = 28
        self.guidance_scale = 0.0
        self.num_inference_steps = 4
        self.max_sequence_length = 256
        self.loaded = self._preload_models("FLUX" in PRELOAD_MODELS)

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

            self.base = FluxPipeline.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16
            )
            # distributed_state = PartialState()
            # self.base.to(distributed_state.device)
            self.base.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
            # self.base = self.base.to(self.device)
            # Optionally compile the base unet
            # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
            print("Base model loaded.")
            return

    def run_flux_schnell_text2img(self, prompt: str):
        if not self.loaded:
            self.loaded = self._preload_models(True)

        # Run both experts
        image = self.base(
            prompt=prompt,
            guidance_scale=self.guidance_scale,
            output_type="pil",
            num_inference_steps=self.num_inference_steps,
            max_sequence_length=self.max_sequence_length,
            # generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        # Convert the image to a format that can be saved to a buffer
        if isinstance(image, torch.Tensor):
            image = Image.fromarray((image * 255).cpu().numpy().astype('uint8'))

        return image
