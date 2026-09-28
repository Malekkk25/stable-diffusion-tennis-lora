# Tennis LoRA: Fine-tuning Stable Diffusion 1.5

A Google Colab notebook that trains a **LoRA** (Low-Rank Adaptation) on top of **Stable Diffusion v1.5** to generate images in a tennis style. The whole workflow is in one notebook: dataset download and filtering, automatic captioning with CLIP, LoRA training with Kohya `sd-scripts`, inference, and a Gradio demo.

*Generated with the prompt `sks tennis style a tennis player hitting a forehand shot` after a short 300-step training run (15 images). Results improve with more images and more steps.*

<img width="512" height="512" alt="sample_output" src="https://github.com/user-attachments/assets/aafce37e-052d-436d-a2f2-8303b7c66b8b" />
## Pipeline

| Step | What happens |
|------|--------------|
| 1. Configuration | Detects the GPU and picks a lightweight training config (T4-friendly) |
| 2. Dataset | Downloads the [Tennis Player Actions dataset](https://www.kaggle.com/datasets/orvile/tennis-player-actions-dataset) (2,000 images) with `kagglehub`, then filters by size, aspect ratio and sharpness (Laplacian variance) and keeps the best images |
| 3. Preprocessing | Center-crops and resizes images to 512×512 |
| 4. Captioning | Uses **CLIP** zero-shot classification to pick the tennis action in each image (forehand, backhand, serve, smash, volley...) and writes a caption such as `sks tennis style a tennis player serving the ball` |
| 5. Dataset layout | Arranges files in the Kohya folder format (`10_sks tennis style/`) |
| 6. Training | Trains the LoRA with Kohya `sd-scripts` (`train_network.py`) |
| 7. Inference | Loads the LoRA into a `diffusers` pipeline and generates a test image |
| 8. Demo | Gradio interface with prompt, negative prompt, steps, guidance, **LoRA strength** and seed controls |

## Training configuration (GPU / T4)

| Parameter | Value |
|-----------|-------|
| Base model | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Training images | 15 |
| Repeats | 10 |
| Resolution | 512 |
| Steps | 300 (about 3.5 minutes on a T4) |
| Batch size | 1 |
| Learning rate | 1e-4 |
| Network dim / alpha | 16 / 8 |
| Precision | fp16 |
| Final average loss | about 0.125 |

The values are in the `CFG` dictionary in the first cell. Increase `target_count` and `max_train_steps` for better quality.

## Tech stack

- Stable Diffusion 1.5, `diffusers`, `transformers`, `peft`
- CLIP (`openai/clip-vit-base-patch32`) for captioning
- Kohya `sd-scripts` for LoRA training
- OpenCV and Pillow for image filtering and preprocessing
- Gradio for the demo
- Google Colab (T4 GPU)

## Getting started

1. Open `TENNIS_DIFFUSION_MODEL.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Set the runtime to **T4 GPU** (Runtime → Change runtime type).
3. Run the cells in order.
4. The trained weights are saved to `/content/work/lora_output/last.safetensors`. Download them from Colab before the session ends.

Notes:

- The dataset comes from Kaggle through `kagglehub`. If the download asks for credentials, add your Kaggle API key in Colab.
- Cell 16 restarts the runtime on purpose after pinning package versions. After the restart, continue from the inference cell.
- `demo.launch(share=True)` creates a temporary public Gradio link that lasts about a week.

## Use the LoRA

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
pipe.load_lora_weights("lora_output", weight_name="last.safetensors")

image = pipe("sks tennis style a tennis player serving the ball",
             num_inference_steps=25).images[0]
image.save("output.png")
```

The trigger phrase is **`sks tennis style`**.

## Limitations

- Trained on only 15 images for 300 steps, so it is a quick proof of concept rather than a polished style model.
- CLIP captions choose from a fixed list of 9 actions, so they are coarse.

## Author

**Malek** ([@Malekkk25](https://github.com/Malekkk25))
