#!/usr/bin/env python3
"""
π0.7 World Model — Baseline (No Optimizations)

Generates subgoal images from input camera images + text instruction.
Uses BAGEL's 3-branch CFG with standard flash attention (no SageAttention),
no quantization, no tensor parallelism.

Input:  N camera images + 1 text instruction
Output: N subgoal images (one per input image)

Usage:
    cd /home/sr5/kunal.swami/eswar/worldModels/Bagel
    python baseline.py
"""

import os
import time
import random
from copy import deepcopy

import numpy as np
import torch
from PIL import Image

# BAGEL imports
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer


# USER CONFIGURATION — Edit these variables
INPUT_IMAGE_PATHS = [
    "./droid_samples/ep001_front.png",
    "./droid_samples/ep001_left.png",
    "./droid_samples/ep001_right.png",
]
INSTRUCTION = "Pick up the candy bar"
MODEL_PATH = "models/BAGEL-7B-MoT"
OUTPUT_DIR = "./pi07_outputs"

#(paper defaults)
NUM_TIMESTEPS = 25      
IMAGE_SHAPE = (384, 512)  

# BAGEL defaults
CFG_TEXT_SCALE = 4.0       
CFG_IMG_SCALE = 2.0        
TIMESTEP_SHIFT = 3.0      
 
SEED = 42                 

def set_seed(seed):
    """Set random seeds for reproducibility (from app.py)."""
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_bagel_model(model_path):
    """Load BAGEL model (standard, no optimizations)."""
    print(f"[Load] Loading from: {model_path}")
    t0 = time.time()

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True, 
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70, 
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2, 
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vit_transform = ImageTransform(448, 336, 14)
    vae_transform = ImageTransform(512, 384, 16)

    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        'language_model.model.embed_tokens', 
        'time_embedder', 
        'latent_pos_embed',
        'vae2llm', 
        'llm2vae', 
        'connector', 
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    # Load Weights
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        offload_folder="offload",
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()

    # Inferencer
    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=vae_transform, vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    print(f"[Load] Done in {time.time()-t0:.1f}s")
    return model, inferencer


# Subgoal Image Generation

@torch.no_grad()
def generate_subgoal_images(inferencer, observations, instruction,
                             num_timesteps=25, cfg_text_scale=4.0,
                             cfg_img_scale=2.0, image_shape=(384, 512)):
    """
    Generate subgoal images using BAGEL's 3-branch CFG.

    For each camera view, generates a subgoal image by:
      1. Setting up 3 CFG branches:
         - gen_context (+text+img): full conditioning
         - cfg_text_context (-text+img): images only, no text
         - cfg_img_context (+text-img): text only, no images
      2. Adding ALL camera images to gen_context and cfg_text_context only
      3. Generating one subgoal per camera (sequential)

    Args:
        observations: dict of {camera_name: PIL_image}
        instruction: text instruction
        num_timesteps: denoising steps (paper: 25)
        cfg_text_scale: text CFG scale (BAGEL default: 4.0)
        cfg_img_scale: image CFG scale (BAGEL default: 2.0)
        image_shape: output image shape (H, W)  paper: (384, 512)

    Returns:
        dict of {camera_name: PIL_image} subgoal images
    """
    subgoal_images = {}

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        for cam_name, cam_img in observations.items():
            # Initialize 3 CFG branches for each camera
            gen_context = inferencer.init_gen_context()          # +text +img
            cfg_text_context = deepcopy(gen_context)              # -text +img
            cfg_img_context = deepcopy(gen_context)               # +text -img

            # Add text to branches 1 and 3 (not branch 2)
            gen_context = inferencer.update_context_text(instruction, gen_context)
            cfg_img_context = inferencer.update_context_text(instruction, cfg_img_context)

            # Add ALL camera images to gen_context and cfg_text_context only
            # (NOT cfg_img_context — it measures impact of images, so must NOT have them)
            # This matches interleave_inference() in inferencer.py
            for cn, ci in observations.items():
                ci_rgb = pil_img2rgb(ci)
                ci_resized = inferencer.vae_transform.resize_transform(ci_rgb)  # Resize like interleave_inference
                # Branch 1 (+text+img): image via VAE + ViT
                gen_context = inferencer.update_context_image(ci_resized, gen_context, vae=True, vit=True)
                # Branch 2 (-text+img): image via VAE + ViT (no text)
                cfg_text_context = inferencer.update_context_image(ci_resized, cfg_text_context, vae=True, vit=True)
                # Branch 3 (+text-img): NO images added (only has text)
                # cfg_img_context stays without images — this is correct per interleave_inference()

            # Generate subgoal image for this camera
            subgoal_img = inferencer.gen_image(
                image_shape=image_shape,
                gen_context=gen_context,
                cfg_text_precontext=cfg_text_context,
                cfg_img_precontext=cfg_img_context,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=[0.0, 1.0],
                timestep_shift=TIMESTEP_SHIFT,
                num_timesteps=num_timesteps,
                cfg_renorm_min=0.0,
                cfg_renorm_type="global",
                enable_taylorseer=False,
            )
            subgoal_images[cam_name] = subgoal_img

    return subgoal_images


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("π0.7 World Model — Baseline (No Optimizations)")
    print("=" * 60)
    print(f"  Input images:  {len(INPUT_IMAGE_PATHS)}")
    print(f"  Instruction:   '{INSTRUCTION}'")
    print(f"  Timesteps:      {NUM_TIMESTEPS}")
    print(f"  Output shape:   {IMAGE_SHAPE[0]}×{IMAGE_SHAPE[1]} (H×W)")
    print(f"  GPUs:           {torch.cuda.device_count()}")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model...")
    model, inferencer = load_bagel_model(MODEL_PATH)

    # Load input images into observations dict
    print("\n[2/3] Loading input images...")
    observations = {}
    for path in INPUT_IMAGE_PATHS:
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found!")
            return
        cam_name = os.path.splitext(os.path.basename(path))[0]
        observations[cam_name] = Image.open(path).convert("RGB")
        print(f"  Loaded: {path} → '{cam_name}'")

    # Generate subgoal images
    print(f"\n[3/3] Generating {len(observations)} subgoal images...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Warmup (5 steps)
    print("  Warmup (5 steps)...")
    set_seed(SEED)
    _ = generate_subgoal_images(inferencer, observations, INSTRUCTION, num_timesteps=5)
    torch.cuda.synchronize()

    # Timed generation (reset seed so warmup doesn't affect reproducibility)
    set_seed(SEED)
    torch.cuda.synchronize()
    start = time.time()

    subgoals = generate_subgoal_images(
        inferencer, observations, INSTRUCTION,
        num_timesteps=NUM_TIMESTEPS,
        cfg_text_scale=CFG_TEXT_SCALE,
        cfg_img_scale=CFG_IMG_SCALE,
        image_shape=IMAGE_SHAPE,
    )

    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Save images
    for cam_name, img in subgoals.items():
        out_path = os.path.join(OUTPUT_DIR, f"{cam_name}_subgoal.png")
        img.save(out_path)
        print(f"  Saved: {out_path}")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Generation time: {elapsed:.2f} seconds")
    print(f"  Subgoals:        {list(subgoals.keys())}")
    print(f"  Paper target:    1.25 seconds (4×H100, INT8, SageAttention, 25 steps)")
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {used:.1f}GB / {total:.1f}GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
