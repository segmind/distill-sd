# distill-sd
![image](https://github.com/segmind/distill-sd/assets/95531133/01ca236a-e616-4049-a043-6a9fdab244bf)

Knowledge-distilled, smaller versions of Stable Diffusion as described in [BK-SDM](https://arxiv.org/abs/2305.15798).<br>
These distillation-trained models produce images of similar quality to the full-sized Stable-Diffusion model while being significantly faster and smaller.<br>
## Components of this Repository:
+ The **[BKSDM directory](/BKSDM)** contains a function to configure the U-net and remove the appropriate blocks prior to distillation training.
+ **[data.py](/data.py)** contains scripts to download data for training. 
+ **[trainT2I.py](/trainT2I.py)** trains the U-net using the methods described in the paper. This might need additional configuration depending on what model type you want to train (base/small/tiny),batch size, hyperparameters etc. 
The basic training code was sourced from the [Huggingface 🤗 diffusers library](https://github.com/huggingface/diffusers).<br>

## Training Details:
Knowledge-Distillation training a neural network is akin to a teacher guiding a student step-by-step (a somewhat loose example). A large teacher model is trained on large amounts of data and then a smaller model is trained on a smaller dataset, with the objective of aping the outputs of the larger model along with classical training on the dataset.<br>
For the Knowledge-Distillation training, we used [SG161222/Realistic_Vision_V4.0's](SG161222/Realistic_Vision_V4.0) U-net  as the teacher model with a subset of [recastai/LAION-art-EN-improved-captions](https://huggingface.co/datasets/recastai/LAION-art-EN-improved-captions) as training data.<br> 


As described in the paper, the final training loss is the sum of the MSE loss between the noise predicted by the teacher U-net and the noise predicted by the student U-net, the MSE Loss between the actual added noise and the predicted noise, and the sum of MSE Losses between the predictions of the teacher and student U-nets after every block.<br>
Total Loss:<br>
![image](https://github.com/segmind/distill-sd/assets/95531133/bf4751cd-99b3-46a9-93e4-d2b4237a9c53)<br>
Task Loss (i.e MSE Loss between added noise and actual noise):<br>
![image](https://github.com/segmind/distill-sd/assets/95531133/86f1d716-97f4-42ad-9e5f-24b091b311eb)<br>
Knowledge Distillation Output Loss (i.e MSE Loss between final output of teacher U-net and student U-net):<br>
![image](https://github.com/segmind/distill-sd/assets/95531133/1b986995-51e6-4c36-bad3-6ca4b719cfd1)<br>
Feature-level Knowledge Distillation Loss (i.e MSE Loss between outputs of each block in the U-net):<br>
![image](https://github.com/segmind/distill-sd/assets/95531133/c5673b95-9e3b-482e-b3bc-a40db6929b5d)<br>
These equations were sourced from the paper.





## Usage 
```python
import torch
from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from torch import Generator


path = 'segmind/BKSDM-Base-45K' # Path to the appropriate model-type
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

## Pretrained checkpoints:
**The trained "base" version of the model is available at [this Huggingface 🤗 repo](https://huggingface.co/segmind/BKSDM-Base-45K).**<br>
**Other versions of the model will be made public soon.**<br>
## Here are some generated examples:
![image](https://github.com/segmind/distill-sd/assets/95531133/8062a175-a042-4a07-9dd7-e35de125f951)

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

