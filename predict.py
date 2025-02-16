# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import torch
from PIL import Image
from typing import List
from load_pipes import (
    OMINICONTROL_WEIGHTS,
    load_depth_processor,
    load_dev_pipe,
    load_schnell_pipe,
)
from src.flux.generate import generate, seed_everything
from src.flux.condition import Condition


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        self.pipe = self.get_pipe()
        load_depth_processor()  # load it here so that it is loaded once at spinup, even though it is used elsewhere.

        print("setup took: ", time.time() - start)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Text prompt",
            default=None,
        ),
        task: str = Input(
            description="OminiControl task",
            default="subject_512",
            choices=OMINICONTROL_WEIGHTS.keys(),
        ),
        control_image: Path = Input(
            description="Control image.",
            default=None,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            ge=1,
            le=50,
            default=8,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for the diffusion process",
            ge=0,
            le=10,
            default=3.5,
        ),
        seed: int = Input(
            description="Random seed. Set for reproducible generation",
            default=None,
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        resolution = 1024 if task == "subject_1024" else 512

        control_image = Image.open(control_image)
        width, height = control_image.size
        assert width == height, "Control image must be a square."
        control_image = control_image.convert("RGB").resize((512, 512))

        condition = Condition(condition_type=task, raw_img=control_image)

        seed_everything(seed)
        images = []
        for _ in range(num_outputs):
            image = generate(
                pipeline=self.pipe,
                prompt=prompt,
                conditions=[condition],
                height=resolution,
                width=resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            images.append(image)

        output_paths = []
        for i, image in enumerate(images):
            output_path = f"./out-{i}.{output_format}"
            if output_format != "png":
                image.save(output_path, quality=output_quality, optimize=True)
            else:
                image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths


class SchnellPredictor(Predictor):
    def get_pipe(self):
        return load_schnell_pipe()


class DevPredictor(Predictor):
    def get_pipe(self):
        return load_dev_pipe()
