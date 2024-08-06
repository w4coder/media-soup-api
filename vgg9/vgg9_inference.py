import torch
from PIL import Image
from torchvision import models, transforms
from vgg9.model import Compiler
from trainer import Trainer
from utils import resize_image, device_checker


class VGG9Processor:
    def __init__(self, device="cuda:0"):
        # Device configuration
        self.device = device_checker(device)
        print(f"vgg9 device : {self.device}")

        # Load VGG19 model
        self.vgg19 = models.vgg19(pretrained=True).features.eval()

        # Define layer names
        self.content_layer_names = ['conv4']
        self.style_layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

        # Transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def load(self):
        print("vgg19 loaded")

    def unload(self):
        print("vgg19 unloaded")

    def process_images(self, content_image: Image.Image, style_image: Image.Image = None):
        # Resize images
        content_image = resize_image(content_image)

        if style_image:
            style_image = resize_image(style_image)
        else:
            # Load default style image
            style_image = Image.open("images/default_style_image.jpg").convert('RGB')
            style_image = resize_image(style_image)

        content_image = self.transform(content_image).unsqueeze(0)

        # Match dimensions
        _, _, target_height, target_width = content_image.shape
        style_aspect_ratio = style_image.width / style_image.height
        target_aspect_ratio = target_width / target_height

        if style_aspect_ratio > target_aspect_ratio:
            new_height = target_height
            new_width = int(style_aspect_ratio * new_height)
        else:
            new_width = target_width
            new_height = int(new_width / style_aspect_ratio)

        resized_style_image = style_image.resize((new_width, new_height), Image.LANCZOS)

        left = (new_width - target_width) / 2
        top = (new_height - target_height) / 2
        right = left + target_width
        bottom = top + target_height

        style_image = resized_style_image.crop((left, top, right, bottom)).resize((target_width, target_height))
        cropped_style_image_tensor = self.transform(style_image).unsqueeze(0)

        compiler = Compiler(self.vgg19, self.content_layer_names, self.style_layer_names, device=self.device)
        model, content_layers, style_layers = compiler.compile(content_image, cropped_style_image_tensor, device=self.device)

        trainer = Trainer(model, content_layers, style_layers, device=self.device)

        input_image = content_image.clone()
        losses, out_image = trainer.fit(input_image, device=self.device)

        return out_image
