
<div align="center">
<img src="https://github.com/segmind/distill-sd/assets/82945616/1f4786fa-5cca-4509-b9c4-5d0f89dd16f9" alt="logo" width="400" height="auto" />
</div>

<div align="center">
<img src="https://github.com/segmind/distill-sd/assets/82945616/8daf90b0-f46d-4cb4-965b-7d443461c4f9" width="950" height="auto" />
</div>


Knowledge-distilled, smaller versions of Stable Diffusion. Unofficial implementation as described in [BK-SDM](https://arxiv.org/abs/2305.15798).<br>
These distillation-trained models produce images of similar quality to the full-sized Stable-Diffusion model while being significantly faster and smaller.<br>
## Components of this Repository:
+ **[data.py](/data.py)** contains scripts to download data for training. 
+ **[distill_training.py](/distill_training.py)** trains the U-net using the methods described in the paper. This might need additional configuration depending on what model type you want to train (sd_small/sd_tiny),batch size, hyperparameters etc. 
The basic training code was sourced from the [Huggingface 🤗 diffusers library](https://github.com/huggingface/diffusers).<br>

## Training Details:
Knowledge-Distillation training a neural network is akin to a teacher guiding a student step-by-step (a somewhat loose example). A large teacher model is trained on large amounts of data and then a smaller model is trained on a smaller dataset, with the objective of aping the outputs of the larger model along with classical training on the dataset.<br>
For the Knowledge-Distillation training, we used [SG161222/Realistic_Vision_V4.0's](SG161222/Realistic_Vision_V4.0) U-net  as the teacher model with a subset of [recastai/LAION-art-EN-improved-captions](https://huggingface.co/datasets/recastai/LAION-art-EN-improved-captions) as training data.<br> 


The final training loss is the sum of the MSE loss between the noise predicted by the teacher U-net and the noise predicted by the student U-net, the MSE Loss between the actual added noise and the predicted noise, and the sum of MSE Losses between the predictions of the teacher and student U-nets after every block.<br>
Total Loss:<br>
![image](https://github.com/segmind/distill-sd/assets/95531133/bf4751cd-99b3-46a9-93e4-d2b4237a9c53)<br>
Task Loss (i.e MSE Loss between added noise and actual noise):<br>
![image](https://github.com/segmind/distill-sd/assets/95531133/86f1d716-97f4-42ad-9e5f-24b091b311eb)<br>
Knowledge Distillation Output Loss (i.e MSE Loss between final output of teacher U-net and student U-net):<br>
![image](https://github.com/segmind/distill-sd/assets/95531133/1b986995-51e6-4c36-bad3-6ca4b719cfd1)<br>
Feature-level Knowledge Distillation Loss (i.e MSE Loss between outputs of each block in the U-net):<br>
![image](https://github.com/segmind/distill-sd/assets/95531133/c5673b95-9e3b-482e-b3bc-a40db6929b5d)<br>

Normal Stable Diffusion U-net:<br>
![image](https://github.com/segmind/distill-sd/assets/95531133/fb1274b4-f81d-44b9-bdfa-72da5ccff519)


SD_Small U-net:<br>
![image](https://github.com/segmind/distill-sd/assets/95531133/7b2ac26a-672f-4218-a055-02278642fa50)

SD_Tiny U-net:<br>
![image](https://github.com/segmind/distill-sd/assets/95531133/0c8cacdd-4267-4373-964e-09820f70e604)









## Usage 
```python
import torch
from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from torch import Generator


path = 'segmind/small-sd' # Path to the appropriate model-type
# Insert your prompt below.
prompt = "Faceshot Portrait of pretty young (18-year-old) Caucasian wearing a high neck sweater, (masterpiece, extremely detailed skin, photorealistic, heavy shadow, dramatic and cinematic lighting, key light, fill light), sharp focus, BREAK epicrealism"
# Insert negative prompt below. We recommend using this negative prompt for best results.
negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck" 

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

# Below code will run on gpu, please pass cpu everywhere as the device and set 'dtype' to torch.float32 for cpu inference.
with torch.inference_mode():
    gen = Generator("cuda")
    gen.manual_seed(1674753452)
    pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    pipe.to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.unet.to(device='cuda', dtype=torch.float16, memory_format=torch.channels_last)

    img = pipe(prompt=prompt,negative_prompt=negative_prompt, width=512, height=512, num_inference_steps=25, guidance_scale = 7, num_images_per_prompt=1, generator = gen).images[0]
    img.save("image.png")
```
## Training the Model:
Training instructions are similar to those of the diffusers text-to-image finetuning script, apart from some extra parameters:<br>
```--distill_level```: One of "sd_small" or "sd_tiny", depending on which type of model is to be trained.<br>
```--output_weight```: A floating point number representing the amount the output-level KD loss is to be scaled by.<br>
```--feature-weight```: A floating point number representing the amount the feautre-level KD loss is to be scaled by.<br>
Also, ```snr_gamma``` has been removed.

An example:<br>
```python
export MODEL_NAME="SG161222/Realistic_Vision_V4.0"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --distill_level="sd_small"\
  --output_weight=0.5\
  --feature_weight=0.5\
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model"
```

## Pretrained checkpoints:
+ The trained "sd-small" version of the model is available at [this Huggingface 🤗 repo](https://huggingface.co/segmind/small-sd)<br>
+ The trained "sd-tiny" version of the model is available at [this Huggingface 🤗 repo](https://huggingface.co/segmind/tiny-sd)<br>
+ Fine-tuned version of the "sd-tiny model" on portrait images is available at [this Huggingface 🤗 repo](https://huggingface.co/segmind/portrait-finetuned)<br>
## Here are some generated examples:
### SD-tiny model fine-tuned on portrait images

Below are some of the images generated with the sd-tiny model, fine-tuned on portrait images.

Link to the model -> [Huggingface 🤗 repo](https://huggingface.co/segmind/portrait-finetuned)

![image](https://github.com/segmind/distill-sd/assets/95531133/84434d4f-06ae-4654-9b94-857210aa16cd)

## Speed comparision of inference on NVIDIA A100 80GB:

![Screenshot from 2023-07-30 22-51-16](https://github.com/segmind/distill-sd/assets/82945616/51dd3fb3-b6fa-429c-9861-61c44e45a171)



![Screenshot from 2023-07-30 22-27-37](https://github.com/segmind/distill-sd/assets/82945616/88b690d6-8fd2-4951-a469-6e2f6b604187)


## Advantages
+ Upto 100% Faster inferences
+ Upto 30% lower VRAM footprint
+ Faster dreambooth and LoRA training

## Limitations
+ The distilled models are in early phase and the outputs may not be at a production quality yet.
+ 

## Citation

```bibtex
@article{kim2023architectural,
  title={On Architectural Compression of Text-to-Image Diffusion Models},
  author={Kim, Bo-Kyeong and Song, Hyoung-Kyu and Castells, Thibault and Choi, Shinkook},
  journal={arXiv preprint arXiv:2305.15798},
  year={2023},
  url={https://arxiv.org/abs/2305.15798}
}
```
```bibtex
@article{Kim_2023_ICMLW,
  title={BK-SDM: Architecturally Compressed Stable Diffusion for Efficient Text-to-Image Generation},
  author={Kim, Bo-Kyeong and Song, Hyoung-Kyu and Castells, Thibault and Choi, Shinkook},
  journal={ICML Workshop on Efficient Systems for Foundation Models (ES-FoMo)},
  year={2023},
  url={https://openreview.net/forum?id=bOVydU0XKC}
}
```

