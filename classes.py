from pydantic import BaseModel
from typing import Literal

class Text2ImgPayload(BaseModel):
    prompt: str
    model: Literal['stabilityai/stable-diffusion-xl-base-1.0', 'stabilityai/stable-diffusion-3-medium-diffusers']

