# distill-sd
distilled diffusion
An unofficial implementation of [BK-SDM](https://arxiv.org/abs/2305.15798).<br>
This distillation-trained U-net produces images of similar quality to the full-sized U-net while being significantly faster.<br>

Includes training scripts for base/small/tiny versions of the distilled models described in the paper.<br>
The trained "base" version of the model is available at [this Huggingface 🤗 repo](https://huggingface.co/segmind/BKSDM-Base-45K).<br>
Other versions of the model will be made public soon.
