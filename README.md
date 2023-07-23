# distill-sd
An unofficial implementation of [BK-SDM](https://arxiv.org/abs/2305.15798).<br>
This distillation-trained U-net produces images of similar quality to the full-sizedStable-Diffusion U-net while being significantly faster.<br>
The BKSDM directory contains a function to configure the U-net and remove the appropriate blocks prior to distillation training.
This code was taken from [this repo](https://github.com/Gothos/BK-SDM).<br>
data.py contains scripts to download data for training. We distillation-train the U-net with [SG161222/Realistic_Vision_V4.0](SG161222/Realistic_Vision_V4.0)  as the teacher model on a subset of [recastai/LAION-art-EN-improved-captions](https://huggingface.co/datasets/recastai/LAION-art-EN-improved-captions).<br> 
trainT2I.py trains the U-net using the methods described in the paper. This might need configuring depending on what model version you want to train. 
The basic training code was sourced from the [🤗 diffusers library](https://github.com/huggingface/diffusers)<br>
The trained "base" version of the model is available at [this Huggingface 🤗 repo](https://huggingface.co/segmind/BKSDM-Base-45K).<br>
Other versions of the model will be made public soon.<br>
## Here are some generated examples:
![image](https://github.com/segmind/distill-sd/assets/95531133/8062a175-a042-4a07-9dd7-e35de125f951)

