import os
import io
import base64
import logging

import torch
import runpod
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

logging.basicConfig(level=logging.INFO)

#MODEL_ID = "Qwen/Qwen-Image-Edit"
MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.info("Loading Qwen Image Edit model...")

# pipe = QwenImageEditPipeline.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
# )
pipe = QwenImageEditPlusPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
)

pipe.to(DEVICE)

logging.info("Model loaded successfully.")


def decode_base64_image(image_string: str) -> Image.Image:
    """
    Convert a base64 string into a PIL Image.
    Supports strings with or without:
    data:image/png;base64,...
    """

    if image_string.startswith("data:image"):
        image_string = image_string.split(",", 1)[1]

    image_bytes = base64.b64decode(image_string)

    image = Image.open(io.BytesIO(image_bytes))

    return image.convert("RGB")


def encode_base64_image(image: Image.Image) -> str:
    """
    Convert PIL Image -> Base64 string.
    """

    buffer = io.BytesIO()

    image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def handler(job):
    try:
        job_input = job["input"]

        if "image" not in job_input:
            return {"error": "Missing 'image' field"}

        if "prompt" not in job_input:
            return {"error": "Missing 'prompt' field"}

        image = decode_base64_image(job_input["image"])

        prompt = job_input["prompt"]

        steps = int(job_input.get("steps", 30))

        guidance = float(job_input.get("guidance_scale", 4.0))

        seed = job_input.get("seed", None)

        generator = None

        if seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        logging.info("Running inference...")

        result = pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=steps,
            true_cfg_scale=guidance,
            generator=generator,
        )

        output_image = result.images[0]

        output_b64 = encode_base64_image(output_image)

        return {
            "success": True,
            "image": output_b64,
        }

    except Exception as e:
        logging.exception("Inference failed")

        return {
            "success": False,
            "error": str(e),
        }


runpod.serverless.start(
    {
        "handler": handler
    }
)
