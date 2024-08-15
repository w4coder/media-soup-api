import io

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal, Optional

from classes import Text2ImgPayload
from models.flux.flux_infernce import FluxSchnellModel
from models.migan.migan_inference import MIGAN_OR
from models.pwrpaint.pwrpt_inference import PowerPaintController
from models.sd3.sd3_infernce import SD3Model
from settings import SDXL_DEVICE, SD3_DEVICE, MIGAN_DEVICE, VGG19_DEVICE, FLUX_DEVICE
from utils import max_column_by_category, rgb_to_grayscale
from models.vgg9.vgg9_inference import VGG9Processor
from models.sdxl.sdxl_inference import SDXLModel

app = FastAPI()

# CORS Configuration
orig_origins = [
    "*",  # Localhost alternative URL
    # Add more origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=orig_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initiate models
vgg9_model = VGG9Processor(
    device=VGG19_DEVICE,
)

# 1. Stable difusion XL
sdxl_model = SDXLModel(
    base_model_path="stabilityai/stable-diffusion-xl-base-1.0",
    refiner_model_path="stabilityai/stable-diffusion-xl-refiner-1.0",
    device=SDXL_DEVICE
)
# 2. Stable difusion 3
sd3_model = SD3Model(
    base_model_path="stabilityai/stable-diffusion-3-medium-diffusers",
    device=SD3_DEVICE,
)

# 3. MIGAN
migan_model = MIGAN_OR(
    device=MIGAN_DEVICE,
)

# 4. PowerPaint model

# pwrpt_model = PowerPaintController(
#     weight_dtype=torch.float16,
#     checkpoint_dir="models/pwrpaint/checkpoints/ppt-v2",
#     local_files_only=True,
#     version="ppt-v2"
# )

# 5. PowerPaint model

flux_schnell_model = FluxSchnellModel(
    base_model_path="black-forest-labs/FLUX.1-schnell",
    device=FLUX_DEVICE,
)

class ModelManager:
    def __init__(self):
        self.models = []

    def register_model(self, model):
        self.models.append(model)

    def unload_all_models(self):
        for model in self.models:
            model.unload()
        self.models.clear()

    def load_new_model(self, new_model, *args, **kwargs):
        self.unload_all_models()
        # new_model = model_class(*args, **kwargs)
        new_model.load()

# Example usage
manager = ModelManager()

manager.register_model(vgg9_model)
manager.register_model(sdxl_model)
manager.register_model(sd3_model)
manager.register_model(migan_model)
# manager.register_model(pwrpt_model)


#### Global instance of the model

@app.post("/vgg9/style-transfer")
async def style_transfer(
        content_image: UploadFile = File(...),
        style_image: UploadFile = File(None)
):
    try:
        # Read image files into PIL Image
        content_image_pil = Image.open(io.BytesIO(await content_image.read())).convert('RGB')
        style_image_pil = Image.open(io.BytesIO(await style_image.read())).convert('RGB') if style_image else None

        manager.load_new_model(vgg9_model)
        # Process images
        output_image = vgg9_model.process_images(content_image_pil, style_image_pil)

        # Convert the output image to a byte array for the response
        output_image = output_image.squeeze(0).cpu()
        output_image = transforms.ToPILImage()(output_image)

        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)

        return StreamingResponse(output_buffer, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/text2img")
async def sdxl_text_to_image(
        payload: Text2ImgPayload,
):
    try:
        if payload.model == "stabilityai/stable-diffusion-xl-base-1.0":
            manager.load_new_model(sdxl_model)
            output_image = sdxl_model.run_sdxl_text2img(prompt=payload.prompt)
        elif payload.model == "stabilityai/stable-diffusion-3-medium-diffusers":
            manager.load_new_model(sd3_model)
            output_image = sd3_model.run_sd3_text2img(prompt=payload.prompt)
        elif payload.model == "black-forest-labs/FLUX.1-schnell":
            manager.load_new_model(flux_schnell_model)
            output_image = flux_schnell_model.run_flux_schnell_text2img(prompt=payload.prompt)
        else:
            raise HTTPException(status_code=400, detail="Unsupported model name")

        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)

        return StreamingResponse(output_buffer, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/remove-object")
async def migan_object_remove(
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        model: Literal[
            'migan-256-places2',
            'migan-512-places2',
            'migan-256-ffhq',
            'comodgan-256-places2',
            'comodgan-512-places2',
            'comodgan-256-ffhq',
            'powerpaintv2'
        ] = Form("powerpaintv2", description="THe model name") ,
        threshold: Optional[float] = Form(7.5, description="Tresshold for object removal"),
        prompt: Optional[str] = Form(None, description="Prompt for object removal"),
        negative_prompt: Optional[str] = Form(None, description="Negative prompt for object removal"),
):
    try:

        if not model:
            raise HTTPException(status_code=400, detail="Model name is required")
        if model not in migan_model.ALLOWED_MODELS:
            raise HTTPException(status_code=400, detail="Unsupported model name")

        model_path = migan_model.ALLOWED_MODELS[model]
        # Read image files into PIL Image
        input_image = Image.open(io.BytesIO(await image.read())).convert("RGB")

        input_mask = Image.open(io.BytesIO(await mask.read())).convert("RGB")
        mask_shape = np.array(input_mask).shape
        input_shape = np.array(input_image).shape
        if "migan" in model or "comodgan" in model:

            # Process images
            output_image = migan_model.process_image(
                img=input_image,
                mask=input_mask,
                model_path=model_path,
                model_name=model
            )
        elif model == "powerpaintv2":
            if not prompt:
                raise HTTPException(status_code=400, detail="A prompt is required to guide the removal process")
            # [output_image], _ = pwrpt_model.infer(
            #     input_image={
            #         "image": input_image,
            #         "mask": input_mask
            #     },
            #     text_guided_prompt=None,
            #     text_guided_negative_prompt=None,
            #     shape_guided_prompt=None,
            #     shape_guided_negative_prompt=None,
            #     fitting_degree=1,
            #     ddim_steps=45,
            #     scale=threshold or 7.5,
            #     seed=1878933855,
            #     task="object-removal",
            #     vertical_expansion_ratio=1,
            #     horizontal_expansion_ratio=1,
            #     outpaint_prompt=None,
            #     outpaint_negative_prompt=None,
            #     removal_prompt=prompt,
            #     removal_negative_prompt=negative_prompt,
            # )
        else:
            raise HTTPException(status_code=400, detail="Unsupported model name")

        # # Convert the output image to a byte array for the response
        # output_image = output_image.squeeze(0).cpu()
        # output_image = transforms.ToPILImage()(output_image)

        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)

        return StreamingResponse(output_buffer, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/image-inpaint")
async def migan_object_remove(
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        model: Literal[
            'powerpaintv2'
        ] = Form("powerpaintv2", description="THe model name"),
        threshold: Optional[float] = Form(7.5, description="Tresshold for object removal"),
        prompt: Optional[str] = Form(None, description="Prompt for object removal"),
        negative_prompt: Optional[str] = Form(None, description="Negative prompt for object removal"),
):
    try:

        if not model:
            raise HTTPException(status_code=400, detail="Model name is required")
        if model not in migan_model.ALLOWED_MODELS:
            raise HTTPException(status_code=400, detail="Unsupported model name")

        # Read image files into PIL Image
        input_image = Image.open(io.BytesIO(await image.read())).convert("RGB")

        input_mask = Image.open(io.BytesIO(await mask.read())).convert("RGB")
        np_mask = np.array(input_mask)

        # Convert the image to grayscale
        grayscale_image = rgb_to_grayscale(np_mask)

        # Convert the grayscale NumPy array back to a PIL Image
        mask_image = Image.fromarray(grayscale_image)
        # Calculate the maximum columns for each category
        max_columns = max_column_by_category(np_mask)

        if model == "powerpaintv2":
            if not prompt:
                raise HTTPException(status_code=400, detail="A prompt is required to guide the removal process")
            # [output_image], _ = pwrpt_model.infer(
            #     input_image={
            #         "image": input_image,
            #         "mask": input_mask
            #     },
            #     text_guided_prompt=prompt,
            #     text_guided_negative_prompt=negative_prompt,
            #     shape_guided_prompt=None,
            #     shape_guided_negative_prompt=None,
            #     fitting_degree=1,
            #     ddim_steps=45,
            #     scale=threshold,
            #     seed=1878933855,
            #     task="text-guided",
            #     vertical_expansion_ratio=1,
            #     horizontal_expansion_ratio=1,
            #     outpaint_prompt=None,
            #     outpaint_negative_prompt=None,
            #     removal_prompt=None,
            #     removal_negative_prompt=None,
            # )
        else:
            raise HTTPException(status_code=400, detail="Unsupported model name")

        # # Convert the output image to a byte array for the response
        # output_image = output_image.squeeze(0).cpu()
        # output_image = transforms.ToPILImage()(output_image)

        output_buffer = io.BytesIO()
        # output_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)

        return StreamingResponse(output_buffer, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
