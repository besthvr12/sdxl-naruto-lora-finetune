# SDXL Naruto Style Fine-tuning

A memory-efficient implementation for fine-tuning Stable Diffusion XL on the Naruto art style dataset using LoRA plus supporting optimizations that fit within Google Colab's free-tier T4 GPU (16GB VRAM).

## Project Structure

```
sdxl-naruto-lora-finetune/
- train.py                    # Main training script
- train_notebook.ipynb        # Training walkthrough (notebook)
- inference_notebook.ipynb    # Inference and before/after comparison
- readme.md                   # This file
```

## High-Level Approach

Fine-tune SDXL for the Naruto style on limited VRAM. Full-model training at 1024x1024 exceeds 16GB, so the solution uses LoRA adapters plus a stack of memory optimizations to keep quality while staying within budget.

## Optimization Summary

1. LoRA adapters on the UNet to train only a small parameter set.
2. Train at 512x512 resolution instead of 1024x1024.
3. Gradient checkpointing to trade compute for memory.
4. 8-bit Adam optimizer to shrink optimizer state.
5. Gradient accumulation to simulate a larger batch size.
6. XFormers memory-efficient attention when available.
7. Keep the VAE in FP32 for stability while freezing it.
8. Freeze text encoders, use gradient clipping, and save frequent checkpoints.

## Resource-Efficient Techniques Explained

### 1. LoRA (Low-Rank Adaptation)

**What it is:** Freeze the base model and inject trainable low-rank matrices into attention layers so only a small set of weights is updated.

**Why I chose it:**
- Reduces trainable parameters to a small fraction of the full model.
- Lowers gradient and optimizer memory.
- Works well for style transfer.

**Implementation:**
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
)
unet = get_peft_model(unet, lora_config)
```

**How it works:** Adds a low-rank update `W + BA` to each targeted weight matrix. The rank `r` controls capacity with minimal memory overhead.

**Observed in log:** 23,224,320 trainable parameters (~0.90% of SDXL) after LoRA injection.

---

### 2. Reduced Resolution Training (512x512)

**What it is:** Train at 512x512 instead of SDXL's native 1024x1024.

**Why I chose it:**
- Cuts activation and latent sizes roughly by a factor of four.
- Makes forward/backward passes fit in 16GB.
- Adequate resolution for style learning in this project.

**Implementation:**
```python
resolution = 512
transforms.Resize(resolution)
transforms.CenterCrop(resolution)
```

---

### 3. Gradient Checkpointing

**What it is:** Recompute intermediate activations during backprop to avoid storing them.

**Why I chose it:**
- Essential for deep UNet blocks to fit memory.
- Acceptable compute trade-off on T4.

**Implementation:**
```python
unet.enable_gradient_checkpointing()
```

---

### 4. 8-bit Adam Optimizer

**What it is:** Quantize optimizer state (moments) to 8-bit.

**Why I chose it:**
- Shrinks optimizer memory with negligible quality loss.
- Critical when only adapters are trainable but still numerous.

**Implementation:**
```python
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=5e-5)
```

---

### 5. Gradient Accumulation

**What it is:** Accumulate gradients across micro-batches before stepping.

**Why I chose it:**
- Effective batch size of 4 with a physical batch size of 1.
- Improves gradient quality without extra VRAM.

**Implementation:**
```python
gradient_accumulation_steps = 4
```

**How it works:** Run forward/backward four times, accumulate gradients, then apply one optimizer step (effective batch size = 4).

---

### 6. XFormers Memory-Efficient Attention

**What it is:** A memory-optimized attention kernel.

**Why I chose it:**
- Attention is a major memory hotspot.
- Reduces attention memory by roughly 40-60% when available.

**Implementation:**
```python
unet.enable_xformers_memory_efficient_attention()
```

---

### 7. VAE in Full Precision (FP32)

**What it is:** Run the VAE in FP32 while keeping it frozen.

**Why I chose it:**
- VAE is numerically sensitive; FP32 prevents NaN/Inf latents.
- No training overhead because weights are frozen.

**Implementation:**
```python
vae.to(accelerator.device, dtype=torch.float32)
vae.requires_grad_(False)
```

---

### 8. Additional Optimizations

- Freeze both text encoders to save optimizer memory:
  ```python
  text_encoder_one.requires_grad_(False)
  text_encoder_two.requires_grad_(False)
  ```
- Gradient clipping for stability:
  ```python
  accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm=0.5)
  ```
- Frequent checkpointing to avoid work loss on Colab interruptions.

---

## Memory Budget (Qualitative)

Fits within 16GB on a T4 because:
- Only LoRA adapters carry gradients and optimizer state; base SDXL weights stay frozen.
- 8-bit Adam shrinks optimizer footprint.
- 512x512 resolution and gradient checkpointing reduce activations.
- FP32 VAE and frozen text encoders add minimal overhead since they are not trained.

---

## Training Configuration

```python
pretrained_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
dataset_name = "lambdalabs/naruto-blip-captions"
resolution = 512
train_batch_size = 1
gradient_accumulation_steps = 4  # Effective batch size: 4
max_train_steps = 1000
learning_rate = 5e-5
lora_rank = 16
lora_alpha = 32
gradient_checkpointing = True
use_8bit_adam = True
max_grad_norm = 0.5
checkpointing_steps = 200
```

### Hyperparameter Choices

- Learning rate 5e-5: conservative for stable LoRA training.
- LoRA rank 16: balances expressiveness vs memory/overfitting.
- LoRA alpha 32: 2x rank for proper scaling.
- Max steps 1000: sufficient for style convergence on this dataset.
- Gradient norm 0.5: strong clipping for stability.
- Mixed precision: disabled (FP32 run) per training log.

---

## Training Process

1. Dataset: 1,221 images from `lambdalabs/naruto-blip-captions` (per training log).
2. Preprocessing: Resize + Center crop + Random horizontal flip + Normalize.
3. Training loop:
   - Encode images to latents (512x512 -> 64x64x4).
   - Add noise per diffusion schedule.
   - Encode text prompts with dual CLIP encoders.
   - Predict noise with UNet (LoRA adapters trainable).
   - Compute MSE loss against actual noise.
   - Backpropagate and update only the adapters.
4. Checkpoints every 200 steps to guard against interruptions (saved at 200, 800, and final 1000 -> `./sdxl-naruto-lora`).

---

## Observed Behavior

With 1000 steps (~2-3 epochs on the dataset):
- Loss decreases steadily without NaN/Inf, aided by FP32 VAE and clipping.
- Naruto style appears after a few hundred steps (anime look, outlines, palette).
- Good generalization: applies style to prompts beyond the training captions while keeping SDXL's general prompt understanding.

---

## Training Log Highlights

- Dataset size: 1,221 images; gradient accumulation 4 -> ~306 update steps per epoch; four epochs to reach 1,000 steps.
- Trainable parameters after LoRA: 23,224,320 (~0.90% of SDXL).
- Runtime (T4): started 05:27:39, finished 07:25:38 (~1h58m wall-clock); mixed precision off (FP32), XFormers enabled, 8-bit Adam used.
- Checkpoints: saved at steps 200, 800, and final 1,000 to `./sdxl-naruto-lora`.
- Final loss: 0.194479 (step 1,000 in `training.log`).

## Usage

### Training
```bash
python train.py
```

### Inference (example)
```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

pipe.load_lora_weights("./sdxl-naruto-lora-finetune")

prompts = [
    "Naruto Uzumaki eating ramen",
    "Bill Gates in Naruto style",
    "A boy with blue eyes in Naruto style",
]

for prompt in prompts:
    image = pipe(prompt, num_inference_steps=30).images[0]
    image.save(f"{prompt[:20]}.png")
```

### Notebook comparison

In `inference_notebook.ipynb`, the first output cell shows the baseline Stable Diffusion XL , and the second output shows the result after our LoRA training.

---

## Monitoring and Debugging

- Track loss: should decrease, not explode or flatline.
- Watch for NaN/Inf in loss or gradients.
- Keep GPU memory under ~15GB.
- Training speed target: ~2-3 seconds per step on T4.

**Common fixes:**
- OOM: keep batch size at 1, ensure gradient checkpointing is on.
- NaN loss: confirm VAE is FP32, lower learning rate.
- Slow training: enable XFormers.
- Poor results: train longer (2000+ steps) or raise LoRA rank.

---

## Requirements

```
torch>=2.0.0
diffusers>=0.21.0
transformers>=4.30.0
accelerate>=0.20.0
peft>=0.4.0
bitsandbytes>=0.41.0
xformers>=0.0.20  # Optional but recommended
datasets>=2.14.0
torchvision>=0.15.0
Pillow>=9.5.0
```

---

## Key Learnings

1. Memory is the bottleneck; optimization is mostly memory management.
2. LoRA is effective: a small parameter fraction learns complex styles.
3. 512px training is a workable compromise for style transfer on tight VRAM.
4. Conservative hyperparameters save runs from instability.
5. Checkpointing often avoids losing long runs on volatile hardware.

---

## Future Improvements

1. Dynamic resolution: mix 512 + 768 + 1024.
2. Mixed precision: re-enable FP16 with careful loss scaling.
3. Advanced LoRA variants: LoKr, LoHa, DoRA.
4. Larger ranks: try r=32 or r=64 for more capacity.
5. Multi-GPU: scale out for speed.
6. Timestep sampling: stratified schedules for better coverage.
7. Offset noise: improved handling of dark/light images.

---

## Notes

- Training time: ~1h55-2h on a T4 for 1,000 steps (per training log).
- Final LoRA size: ~70MB; easy to share or combine with other LoRAs.
- Works with standard SDXL inference pipelines.

---

## Acknowledgments

- Stability AI for SDXL base model.
- Lambda Labs for the Naruto dataset.
- Hugging Face for the diffusers stack.
- Microsoft for the LoRA technique.
- bitsandbytes team for 8-bit optimization.

---

## License

This project follows the licenses of its dependencies:
- SDXL: CreativeML Open RAIL++-M License.
- Dataset: Check `lambdalabs/naruto-blip-captions` license.

---

**Author:** Harsh Verma  
**Contact:** harshv034@gmail.com  
**Date:** December 2025
