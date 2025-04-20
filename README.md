import os
import io
import requests
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

def download_lora(lora_url):
    local_path = "/tmp/lora.safetensors"
    r = requests.get(lora_url)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(r.content)
    return local_path

def predict(prompt: str, lora_url: str = None):
    # Load the base Flux model from Hugging Face
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",  # Replace with actual Flux path if needed
        torch_dtype=torch.float16
    ).to("cuda")

    if lora_url:
        lora_path = download_lora(lora_url)
        pipe.load_lora_weights(lora_path)

    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

    # Return image as bytes
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
