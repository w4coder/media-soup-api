<br/>
<div align="center">
   <a href="https://github.com/w4coder/MGMclient">
    <img src="/assets/logo.svg" alt="Logo" width="80" height="80">
   </a>
</div>
<br/>

# Stable Diffusion API & Image processing tools API

This project provides a set of image processing services using various models, including style transfer, text-to-image generation, and object removal. The application is built with FastAPI and utilizes several image processing models.

## Features

- **Style Transfer**: Apply style transfer using the VGG9 model.
- **Text-to-Image Generation**: Generate images from text prompts using Stable Diffusion models.
- **Object Removal**: Remove objects from images using MIGAN and PowerPaint models.
- **Image Inpainting**: Fill missing parts of images using the PowerPaint model.

## Requirements

- Python 3.8 or higher
- Required Python packages listed in `requirements.txt`

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
    ``` 
2. **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
Download pre-trained MI-GAN models from [here](https://drive.google.com/drive/folders/1xNtvN2lto0p5yFKOEEg9RioMjGrYM74w?usp=share_link) and put into `./models/migan` directory.
If you also want to test with Co-Mod-GAN models, download pre-trained models from [here](https://drive.google.com/drive/folders/1VATyNQQJW2VpuHND02bc-3_4ukJMHQ44?usp=share_link) and put into `./models` directory.

## Running the Application

To run the FastAPI application locally, use the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

The server will be available at http://localhost:5000.

## API Endpoints
1. Style Transfer

    * Endpoint: `/vgg9/style-transfer`\
    * Method: POST\
    * Parameters:\
        * content_image (required): Image file to apply style to.\
        * style_image (optional): Image file with the style. 

2. Text-to-Image Generation

    * Endpoint: /text2img\
    * Method: POST\
    * Parameters:\
        * model (required): The model to use (stabilityai/stable-diffusion-xl-base-1.0 or stabilityai/stable-diffusion-3-medium-diffusers).\
        * prompt (required): Text prompt for image generation.

3. Object Removal

    * Endpoint: /remove-object\
    * Method: POST\
    * Parameters:\
        * image (required): Image file with objects to be removed.\
        * mask (required): Mask file indicating objects to be removed.\
        * model (optional): Model to use for object removal.\
          *  migan-256-places2: **good**
          *  migan-512-places2: **good**
          *  migan-256-ffhq: **good**
          *  comodgan-256-places2: **good**
          *  comodgan-512-places2: **good**
          *  comodgan-256-ffhq: **good**
          *  powerpaintv2: **sometimes unacurate due to difusion model version**
        * threshold (optional): Threshold for object removal.\
        * prompt (optional): Prompt for object removal.\
        * negative_prompt (optional): Negative prompt for object removal.
    
4. Image Inpainting

    * Endpoint: /image-inpaint\
    * Method: POST\
    * Parameters:\
        * image (required): Image file with areas to be inpainted.\
        * mask (required): Mask file indicating areas to be inpainted.\        model (required): Model to use for inpainting.
        * threshold (optional): Threshold for inpainting.\
        * prompt (optional): Prompt for inpainting.\
        * negative_prompt (optional): Negative prompt for inpainting.

Upgrading Models

You can upgrade to your own models by modifying the ModelManager class in the main.py file:

    Register Your Model:
    Add your model initialization in the ModelManager class.

    Load New Model:
    Use the manager.load_new_model(new_model, *args, **kwargs) method to load your model.

Refer to the specific model class documentation for details on how to initialize and use different models.
Loading Stable Diffusion Models from Hugging Face

To use Stable Diffusion models from Hugging Face:

    Create an Account:
    Go to Hugging Face and sign up for an account.

    Agree to Model Terms:
    Navigate to the model page:
        Stable Diffusion XL and click on "Access Repository".
        Stable Diffusion 3 and click on "Access Repository".

    Obtain Your Token:
    Once you have agreed to the terms, obtain your access token from your Hugging Face account settings and update your model paths in the main.py file with the appropriate model names and paths.

Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.
License

This project is licensed under the MIT License. See the LICENSE file for details.

sql


Feel free to adjust the URLs, repository names, and any specific details to match your actual project
