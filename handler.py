import os
import uuid
import torch
import runpod

from diffusers import QwenImageEditPipeline

from utils import load_image

OUTPUT_DIR = "/tmp"

print("Loading model...")

pipe = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=torch.bfloat16
)

pipe.to("cuda")

print("Model Ready")


def handler(job):

    job_input = job["input"]

    image = load_image(job_input["image"])

    prompt = job_input["prompt"]

    steps = job_input.get("steps", 30)

    cfg = job_input.get("cfg", 4)

    generator = torch.Generator("cuda").manual_seed(
        job_input.get("seed", 42)
    )

    result = pipe(
        image=image,
        prompt=prompt,
        num_inference_steps=steps,
        true_cfg_scale=cfg,
        generator=generator
    )

    filename = f"{uuid.uuid4()}.png"

    path = os.path.join(OUTPUT_DIR, filename)

    result.images[0].save(path)

    return {
        "image_path": path
    }


runpod.serverless.start(
    {
        "handler": handler
    }
)
