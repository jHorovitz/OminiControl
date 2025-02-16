import os
import time
import torch
import subprocess
from image_gen_aux import DepthPreprocessor
from diffusers import FluxPipeline

DEVICE = torch.device("cuda")
WEIGHT_DTYPE = torch.bfloat16


SCHNELL_PIPELINE = None
DEV_PIPELINE = None
DEPTH_PROCESSOR = None

MODEL_CACHE_TOP_DIR = "./model-cache"  # necessary for tars that also contain a directory.
SCHNELL_CACHE = "./model-cache/FLUX.1-schnell"
SCHNELL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/files.tar"
DEV_CACHE = "./model-cache/FLUX.1-dev"
DEV_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
DEPTH_PROCESSOR_CACHE = (
    "./model-cache/models--LiheYoung--depth-anything-large-hf/snapshots/27ccb0920352c0c37b3a96441873c8d37bd52fb6"
)
DEPTH_PROCESSOR_URL = "https://weights.replicate.delivery/default/redux-slider/LiheYoung/depth-anything-large-hf.tar"


OMINICONTROL_WEIGHTS = {
    "subject_512": "ominicontrol_weights/omini/subject_512.safetensors",
    "subject_1024": "ominicontrol_weights/omini/subject_1024_beta.safetensors",
    "fill": "ominicontrol_weights/experimental/fill.safetensors",
    "canny": "ominicontrol_weights/experimental/canny.safetensors",
    "depth": "ominicontrol_weights/experimental/depth.safetensors",
    "coloring": "ominicontrol_weights/experimental/coloring.safetensors",
    "deblurring": "ominicontrol_weights/experimental/deblurring.safetensors",
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def load_flux_with_loras(model_cache, model_url):
    if not os.path.exists(model_cache):
        download_weights(model_url, MODEL_CACHE_TOP_DIR)

    pipe = FluxPipeline.from_pretrained(
        model_cache,
        torch_dtype=WEIGHT_DTYPE,
        local_files_only=True,
    ).to(DEVICE)

    for name, path in OMINICONTROL_WEIGHTS.items():
        pipe.load_lora_weights(path, weight_name=path, adapter_name=name, cache_dir=".", local_files_only=True)

    return pipe


def load_schnell_pipe():
    global SCHNELL_PIPELINE
    if SCHNELL_PIPELINE is None:
        print("Loading Flux schnell pipeline")
        pipe = load_flux_with_loras(SCHNELL_CACHE, SCHNELL_URL)
        SCHNELL_PIPELINE = pipe
    return SCHNELL_PIPELINE


def load_dev_pipe():
    global DEV_PIPELINE
    if DEV_PIPELINE is None:
        print("Loading Flux dev pipeline")
        pipe = load_flux_with_loras(DEV_CACHE, DEV_URL)
        DEV_PIPELINE = pipe
    return DEV_PIPELINE


def load_depth_processor():
    global DEPTH_PROCESSOR
    if DEPTH_PROCESSOR is not None:
        return DEPTH_PROCESSOR
    if not os.path.exists(DEPTH_PROCESSOR_CACHE):
        download_weights(DEPTH_PROCESSOR_URL, MODEL_CACHE_TOP_DIR)
    depth_processor = DepthPreprocessor.from_pretrained(DEPTH_PROCESSOR_CACHE)
    DEPTH_PROCESSOR = depth_processor
    return DEPTH_PROCESSOR
