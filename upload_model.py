import torch
from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from PIL import Image
import os
from huggingface_hub import create_repo, upload_folder

def make_image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def save_model_card(
    prompts,
    repo_id: str,
    images=None,
    repo_folder=None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: SG161222/Realistic_Vision_V4.0
datasets:
- recastai/LAION-art-EN-improved-captions
tags:
- bksdm
- bksdm-base
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
inference: true
---
    """
    model_card = f"""
# Text-to-image Distillation - {repo_id}

This pipeline was distilled from **SG161222/Realistic_Vision_V4.0** on a Subset of **recastai/LAION-art-EN-improved-captions** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {prompts}: \n
{img_str}

This Pipeline is based upon [the paper](https://arxiv.org/pdf/2305.15798.pdf). Training Code can be found [here](https://github.com/segmind/BKSDM).

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "Portrait of a pretty girl"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Steps: 5000
* Learning rate: 1e-4
* Batch size: 32
* Gradient accumulation steps: 4
* Image resolution: 512
* Mixed-precision: fp16

"""

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)
        
repo_id = create_repo(
                repo_id="BKSDM-Base-5", exist_ok=True, token="token"
            ).repo_id
prompts = ["Portrait of a pretty girl", "photo of super car, natural lighting, 8k uhd, high quality, film grain, Fujifilm XT3"]
images = []
images.append(Image.open("civit/45K_girl_2.png"))
images.append(Image.open("civit/45K_car_1.png"))
save_model_card(prompts, repo_id, images=images, repo_folder="5K")
upload_folder(
    repo_id=repo_id,
    folder_path="5K",
    commit_message="Initial Commit",
    ignore_patterns=["step_*", "epoch_*"],
)