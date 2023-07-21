import torch
import gc
import diffusers
import transformers


class BKSDM:
    def __init__(
        self,
        model_id="runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        model_type="base",
        device="cuda",
        **kwargs
    ):
        assert model_type in ["base", "tiny", "midless", "small"]
        if not torch.cuda.is_available():
            device = "cpu"
            torch_dtype = torch.float32
        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype
        )
        self.pipe.to(device)
        # Set mid block to None if mode is other than base
        if model_type != "base":
            self.pipe.unet.mid_block = None
        # Commence deletion of resnets/attentions inside the U-net
        if model_type != "midless":
            # Handle Down Blocks
            for i in range(3):
                delattr(self.pipe.unet.down_blocks[i].resnets, "1")
                delattr(self.pipe.unet.down_blocks[i].attentions, "1")

            if model_type == "tiny":
                delattr(self.pipe.unet.down_blocks, "3")
                self.pipe.unet.down_blocks[2].downsamplers = None

            else:
                delattr(self.pipe.unet.down_blocks[3].resnets, "1")
            # Handle Up blocks

            self.pipe.unet.up_blocks[0].resnets[1] = self.pipe.unet.up_blocks[
                0
            ].resnets[2]
            delattr(self.pipe.unet.up_blocks[0].resnets, "2")
            for i in range(1, 4):
                self.pipe.unet.up_blocks[i].attentions[1] = self.pipe.unet.up_blocks[
                    i
                ].attentions[2]
                delattr(self.pipe.unet.up_blocks[i].attentions, "2")
                delattr(self.pipe.unet.up_blocks[i].resnets, "1")
            if model_type == "tiny":
                for i in range(3):
                    self.pipe.unet.up_blocks[i] = self.pipe.unet.up_blocks[i + 1]
                delattr(self.pipe.unet.up_blocks, "3")
        torch.cuda.empty_cache()
        gc.collect()

    def __call__(
        self, prompt, num_inference_steps=50, guidance_scale=7.5, negative_prompt=None
    ):
        return self.pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
        )


def prepare_unet(unet, model_type):
    assert model_type in ["base", "tiny", "midless", "small"]
    # Set mid block to None if mode is other than base
    if model_type != "base":
        unet.mid_block = None
    # Commence deletion of resnets/attentions inside the U-net
    if model_type != "midless":
        # Handle Down Blocks
        for i in range(3):
            delattr(unet.down_blocks[i].resnets, "1")
            delattr(unet.down_blocks[i].attentions, "1")

        if model_type == "tiny":
            delattr(unet.down_blocks, "3")
            unet.down_blocks[2].downsamplers = None

        else:
            delattr(unet.down_blocks[3].resnets, "1")
        # Handle Up blocks

        unet.up_blocks[0].resnets[1] = unet.up_blocks[0].resnets[2]
        delattr(unet.up_blocks[0].resnets, "2")
        for i in range(1, 4):
            unet.up_blocks[i].resnets[1] = unet.up_blocks[i].resnets[2]
            unet.up_blocks[i].attentions[1] = unet.up_blocks[i].attentions[2]
            delattr(unet.up_blocks[i].attentions, "2")
            delattr(unet.up_blocks[i].resnets, "2")
        if model_type == "tiny":
            for i in range(3):
                unet.up_blocks[i] = unet.up_blocks[i + 1]
            delattr(unet.up_blocks, "3")
    torch.cuda.empty_cache()
    gc.collect()
