import numpy as np
import torch
import torch.utils.checkpoint
import os

from tqdm.auto import tqdm
from einops import rearrange
from src.flux.sampling import get_noise, get_schedule, prepare, unpack
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft
from src.flux.model import Flux, FluxParams
from src.flux.controlnet import ControlNetFlux
from src.flux.modules.conditioner import HFEmbedder
from src.flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from dataclasses import dataclass
from PIL import Image

@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None
    repo_id_ae: str | None

configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-fp8": ModelSpec(
        repo_id="XLabs-AI/flux-dev-fp8",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux-dev-fp8.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV_FP8"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}

def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))

def load_flow_model(name: str, device: str | torch.device = "cuda", hf_download: bool = True, ckpt_path=None):
    # Loading Flux
    print("Init model")
    if ckpt_path is None:
        ckpt_path = configs[name].ckpt_path
    if (
            ckpt_path is None
            and configs[name].repo_id is not None
            and configs[name].repo_flow is not None
            and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow.replace("sft", "safetensors"))

    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params)

    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return model

def load_controlnet(name, device, transformer=None):
    with torch.device(device):
        controlnet = ControlNetFlux(configs[name].params).to(device)
    if transformer is not None:
        controlnet.load_state_dict(transformer.state_dict(), strict=False).to(device)
    return controlnet

def load_t5(device: str | torch.device = "cuda", max_length: int = 512,
            version="xlabs-ai/xflux_text_encoders") -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder(version, max_length=max_length, torch_dtype=torch.bfloat16).to(device)

def load_clip(device: str | torch.device = "cuda", version="openai/clip-vit-large-patch14") -> HFEmbedder:
    return HFEmbedder(version, max_length=77, torch_dtype=torch.bfloat16, is_clip=True).to(device)

def load_ae(name: str, device: str | torch.device = "cuda", hf_download: bool = True, ckpt_path=None) -> AutoEncoder:
    if ckpt_path is None:
        ckpt_path = configs[name].ae_path
    if (
            ckpt_path is None
            and configs[name].repo_id is not None
            and configs[name].repo_ae is not None
            and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id_ae, configs[name].repo_ae)

    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta" if ckpt_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return ae

def offload_model_to_cpu(*models, offload):
    if not offload: return
    for model in models:
        model.cpu()
        torch.cuda.empty_cache()

class PixelPonder:
    def __init__(self,
                 controlnet_ckpt_path: str,
                 device: str | torch.device = "cuda",
                 offload: bool = False,
                 dit_ckpt_path: str = None,
                 t5_ckpt_path: str = None,
                 clip_ckpt_path: str = None,
                 ae_ckpt_path: str = None,
                 is_schnell: bool = False,):
        """
            Initialize all components of PixelPonder and load the weights.

            controlnet_ckpt_path: the path of pixelponder's checkpoint
            offload: In the inference process, the weights of components that are temporarily not in use will be moved
                     to the CPU to save GPU memory. This will significantly reduce the inference speed.
            dit_ckpt_path: the path of FLUX's checkpoint. If it is None, it will be automatically downloaded online.
                           If it is set to None, the corresponding weight files will be automatically downloaded from Hugging Face.
            clip_ckpt_path: the path of clip's checkpoint. If it is None, it will be automatically downloaded online.If
                            it is set to None, the corresponding weight files will be automatically downloaded from Hugging Face.
            ae_ckpt_path: the path of ae's checkpoint. If it is None, it will be automatically downloaded online.If it is
                          set to None, the corresponding weight files will be automatically downloaded from Hugging Face.
        """
        self.device = device
        self.offload = offload
        self.dit = load_flow_model("flux-dev", device="cpu" if offload else device,
                                   ckpt_path=dit_ckpt_path if dit_ckpt_path is not None else None)
        self.dit.requires_grad_(False)

        if 'fp32' in controlnet_ckpt_path:
            self.dtype = torch.float32
        else:
            self.dtype = torch.bfloat16
        self.controlnet = load_controlnet(name="flux-dev", device=device)
        self.controlnet = self.controlnet.to(dtype=self.dtype, device=device)
        self.controlnet.load_state_dict(torch.load(controlnet_ckpt_path))
        self.controlnet.requires_grad_(False)

        self.t5 = load_t5(device, max_length=256 if is_schnell else 512, version=t5_ckpt_path if t5_ckpt_path is not None else None)
        self.t5 = self.t5.to(device=device)
        self.t5.requires_grad_(False)
        offload_model_to_cpu(self.t5, offload=offload)

        self.clip = load_clip(device, version=clip_ckpt_path if clip_ckpt_path is not None else None)
        self.clip = self.clip.to(device=device)
        self.clip.requires_grad_(False)

        self.ae = load_ae("flux-dev", device="cpu" if offload else device,
                          ckpt_path=ae_ckpt_path if ae_ckpt_path is not None else None)
        self.ae = self.ae.to(device=device)
        self.ae.requires_grad_(False)
        offload_model_to_cpu(self.controlnet, self.dit, offload=offload)

    def __call__(self,
                 text: str,
                 conditions: dict[str, Image.Image],
                 seed: int = 100,
                 num_steps: int = 25
                 ) -> Image.Image:
        """
            Forward inference

            text : the text description of the target image.
            conditions : the visual conditions of the target image.
            seed : the random seed
            num_steps : sampling steps
        """
        torch.manual_seed(seed)

        hints = {}
        for type, condition in conditions.items():
            width, height = condition.size
            if width != 512 or height != 512:
                raise ValueError(f"Image dimensions do not match: {width}x{height}")
            else:
                hints[type] = (torch.from_numpy((np.array(condition) / 127.5) - 1)).permute(2, 0, 1).to(dtype=self.dtype, device=self.device).unsqueeze(0)

        x = get_noise(
            1, 512, 512, device=torch.device(self.device),
            dtype=torch.bfloat16, seed=seed
        )

        timesteps = get_schedule(
            num_steps=num_steps,
            image_seq_len=(512 // 8) * (512 // 8) // (16 * 16),
            shift=True,
        )

        if self.offload:
            self.t5.to(self.device), self.clip.to(self.device)
        inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=text)
        offload_model_to_cpu(self.t5, self.clip, offload=self.offload)

        img = inp_cond['img'].to(dtype=self.dtype, device=self.device)
        txt = inp_cond['txt'].to(dtype=self.dtype, device=self.device)
        img_ids = inp_cond['img_ids'].to(dtype=self.dtype, device=self.device)
        txt_ids = inp_cond['txt_ids'].to(dtype=self.dtype, device=self.device)
        vec = inp_cond['vec'].to(dtype=self.dtype, device=self.device)
        guidance_vec = torch.full((img.shape[0],), 4.0, dtype=self.dtype, device=self.device)

        # denoise
        for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:]), desc="Denoising", total=len(timesteps[:-1])):
            if self.offload:
                self.controlnet.to(self.device)

            if self.dtype == torch.float32:
                img = img.to(dtype=self.dtype, device=self.device)
                txt = txt.to(dtype=self.dtype, device=self.device)
                img_ids = img_ids.to(dtype=self.dtype, device=self.device)
                txt_ids = txt_ids.to(dtype=self.dtype, device=self.device)
                vec = vec.to(dtype=self.dtype, device=self.device)
                guidance_vec = guidance_vec.to(dtype=self.dtype, device=self.device)

            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

            block_res_samples = self.controlnet(
                img=img,
                img_ids=img_ids,
                controlnet_cond=hints,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            if self.offload:
                offload_model_to_cpu(self.controlnet, offload=self.offload)
                self.dit.to(self.device)

            if self.dtype == torch.float32:
                img = img.to(dtype=torch.bfloat16, device=self.device)
                txt = txt.to(dtype=torch.bfloat16, device=self.device)
                img_ids = img_ids.to(dtype=torch.bfloat16, device=self.device)
                txt_ids = txt_ids.to(dtype=torch.bfloat16, device=self.device)
                vec = vec.to(dtype=torch.bfloat16, device=self.device)
                guidance_vec = guidance_vec.to(dtype=torch.bfloat16, device=self.device)

            pred = self.dit(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                block_controlnet_hidden_states=[
                    0.8 * sample.to(dtype=torch.bfloat16) for sample in block_res_samples
                ],
                y=vec,
                timesteps=t_vec.to(dtype=torch.bfloat16),
                guidance=guidance_vec,
            )
            if self.offload:
                offload_model_to_cpu(self.dit, offload=self.offload)
            img = img + (t_prev - t_curr) * pred
        x = img

        if self.offload:
            offload_model_to_cpu(self.dit, self.controlnet, offload=self.offload)
            self.ae.decoder.to(x.device)

        x = unpack(x.float(), 512, 512)
        x = self.ae.decode(x)

        offload_model_to_cpu(self.ae.decoder, offload=self.offload)

        x = x.squeeze(0)
        x = rearrange(x.clamp(-1, 1), "c h w -> h w c")
        image = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        return image



























