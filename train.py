"""
TRAINING SCRIPT WITH VISIBLE LOSS
Loss prints every single step for debugging
"""

import os
import math
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    pretrained_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    dataset_name = "lambdalabs/naruto-blip-captions"
    resolution = 512
    train_batch_size = 1
    gradient_accumulation_steps = 4
    max_train_steps = 1000
    learning_rate = 5e-5  # Reduced for stability
    lora_rank = 16
    lora_alpha = 32
    mixed_precision = "no"  # Disabled for stability
    gradient_checkpointing = True
    use_8bit_adam = True
    max_grad_norm = 0.5
    output_dir = "./sdxl-naruto-lora"
    checkpointing_steps = 200
    seed = 42
    num_workers = 2
    log_file = "training.log"

config = Config()

def setup_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logger(os.path.join(config.output_dir, config.log_file))

logger.info("TRAINING CONFIGURATION")
logger.info(f"Learning Rate: {config.learning_rate}")
logger.info(f"Mixed Precision: {config.mixed_precision}")
logger.info(f"Max Steps: {config.max_train_steps}")
logger.info(f"Checkpoint Every: {config.checkpointing_steps} steps")

# ============================================================================
# DATASET
# ============================================================================

class NarutoDataset(Dataset):
    def __init__(self, dataset, transform, tokenizer_one, tokenizer_two):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
    
    def __len__(self):
        return len(self.dataset)
    
    def tokenize_caption(self, caption):
        tokens_one = self.tokenizer_one(
            caption, padding="max_length", 
            max_length=self.tokenizer_one.model_max_length,
            truncation=True, return_tensors="pt"
        ).input_ids[0]
        
        tokens_two = self.tokenizer_two(
            caption, padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            truncation=True, return_tensors="pt"
        ).input_ids[0]
        
        return tokens_one, tokens_two
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        image = self.transform(image)
        caption = item["text"]
        tokens_one, tokens_two = self.tokenize_caption(caption)
        
        return {
            "pixel_values": image,
            "input_ids_one": tokens_one,
            "input_ids_two": tokens_two,
            "caption": caption
        }

def get_transforms(resolution):
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

# ============================================================================
# TRAINING
# ============================================================================

def train():
    logger.info("Initializing training...")
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision
    )
    
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    
    logger.info("Loading models...")
    
    # Load all models
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.pretrained_model_name, subfolder="scheduler"
    )
    tokenizer_one = CLIPTokenizer.from_pretrained(
        config.pretrained_model_name, subfolder="tokenizer"
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        config.pretrained_model_name, subfolder="tokenizer_2"
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        config.pretrained_model_name, subfolder="text_encoder"
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        config.pretrained_model_name, subfolder="text_encoder_2"
    )
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name, subfolder="unet"
    )
    
    # Freeze non-trainable models
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    logger.info("Applying LoRA...")
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    
    # Print trainable parameters
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    logger.info(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
    
    if hasattr(unet, "enable_xformers_memory_efficient_attention"):
        unet.enable_xformers_memory_efficient_attention()
        logger.info("XFormers enabled")
    
    # Optimizer
    if config.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=config.learning_rate)
        logger.info("Using 8-bit Adam")
    else:
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
    
    logger.info("Loading dataset...")
    dataset = load_dataset(config.dataset_name, split="train")
    logger.info(f"Dataset: {len(dataset)} images")
    
    transform = get_transforms(config.resolution)
    train_dataset = NarutoDataset(dataset, transform, tokenizer_one, tokenizer_two)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True, 
        num_workers=config.num_workers
    )
    
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)
    
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=100 * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_steps * config.gradient_accumulation_steps
    )
    
    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move models to device - USE FP32 FOR VAE
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device)
    text_encoder_two.to(accelerator.device)
    
    logger.info("STARTING TRAINING")
    logger.info(f"Total epochs: {num_train_epochs}")
    logger.info(f"Steps per epoch (approx): {num_update_steps_per_epoch}")
    logger.info(f"Total steps: {config.max_train_steps}")
    
    # Training loop
    global_step = 0
    
    for epoch in range(num_train_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_train_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Encode images to latents
                with torch.no_grad():
                    pixel_values = batch["pixel_values"]
                    # Clamp to prevent extreme values
                    pixel_values = torch.clamp(pixel_values, -1.0, 1.0)
                    
                    # Encode with FP32 VAE
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()
                
                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                with torch.no_grad():
                    enc_out_1 = text_encoder_one(
                        batch["input_ids_one"],
                        output_hidden_states=True
                    )
                    enc_out_2 = text_encoder_two(
                        batch["input_ids_two"],
                        output_hidden_states=True
                    )
                    
                    prompt_embeds = torch.cat([
                        enc_out_1.hidden_states[-2],
                        enc_out_2.hidden_states[-2]
                    ], dim=-1)
                    
                    pooled_prompt_embeds = enc_out_2.text_embeds
                
                # Prepare time IDs
                add_time_ids = torch.cat([
                    torch.tensor([[
                        config.resolution, config.resolution, 
                        0, 0, 
                        config.resolution, config.resolution
                    ]]) for _ in range(bsz)
                ]).to(latents.device)
                
                # Predict noise
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": add_time_ids
                    }
                ).sample
                
                # Compute loss
                loss = F.mse_loss(
                    model_pred.float(), 
                    noise.float(), 
                    reduction="mean"
                )
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss ({loss.item()}) - skipping batch")
                    continue
                
                # Backpropagation
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    # Clip gradients
                    grad_norm = accelerator.clip_grad_norm_(
                        unet.parameters(), 
                        config.max_grad_norm
                    )
                    
                    # Check gradient norm
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logger.warning("Invalid gradient norm - skipping update")
                        optimizer.zero_grad()
                        continue
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update global step
            if accelerator.sync_gradients:
                global_step += 1
                
                current_loss = loss.item()
                logger.info(
                    f"Step {global_step:4d}/{config.max_train_steps} | Loss: {current_loss:.6f}"
                )
                
                # Save checkpoints
                if global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            config.output_dir, 
                            f"checkpoint-{global_step}"
                        )
                        accelerator.unwrap_model(unet).save_pretrained(save_path)
                        logger.info(f"Checkpoint saved: {save_path}")
                
                # Stop at max steps
                if global_step >= config.max_train_steps:
                    break
        
        if global_step >= config.max_train_steps:
            break
    
    # Final save
    if accelerator.is_main_process:
        final_path = config.output_dir
        accelerator.unwrap_model(unet).save_pretrained(final_path)
        
        logger.info("TRAINING COMPLETE")
        logger.info(f"Final model saved to: {final_path}")
        logger.info(f"Total steps completed: {global_step}")
        logger.info(f"Final loss: {current_loss:.6f}")

if __name__ == "__main__":
    train()
