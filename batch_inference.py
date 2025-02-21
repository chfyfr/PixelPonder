import numpy as np
import torch
import torch.utils.checkpoint
from tqdm.auto import tqdm
from einops import rearrange
from src.flux.sampling import get_noise, get_schedule, prepare, unpack
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft
from src.flux.model import Flux, FluxParams
from src.flux.controlnet import ControlNetFlux
from src.flux.modules.conditioner import HFEmbedder
from src.flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from dataclasses import dataclass
import os
from torch.utils.data import Dataset, DataLoader
import json
import multiprocessing
import random
import argparse


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


def c_crop(image):  # crop center region of image
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


class InfImageDataset(Dataset):
    def __init__(self, input_jsonlines,
                 img_size=512,
                 caption_type='json',
                 hint_types=['canny', 'depth', 'hed'], ):
        self.img_size = img_size
        self.caption_type = caption_type
        self.hint_types = hint_types

        self.input_jsonlines = input_jsonlines

    def __len__(self):
        return len(self.input_jsonlines)

    def __getitem__(self, idx):
        try:
            data = self.input_jsonlines[idx]
            img = Image.open(data['image']).convert('RGB')
            img_name = os.path.basename(data['image'])
            img = c_crop(img)
            img = img.resize((self.img_size, self.img_size))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)

            hints = {}
            for type in self.hint_types:
                hint = Image.open(data[type]).convert('RGB')
                hint = c_crop(hint)
                hint = hint.resize((self.img_size, self.img_size))
                hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
                hint = hint.permute(2, 0, 1)
                hints[type] = hint

            if self.caption_type == 'json':
                prompt = data['text']
            else:
                prompt = data[self.caption_type]

            return img, prompt, hints, img_name
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.input_jsonlines) - 1))


def inf(bs: int, is_schnell: bool, device: str, offload: bool, seed: int, num_steps: int, ckpt_path: str,
        input_jsonlines: list, out_dir: str, selected_hints: dict):
    dataset = InfImageDataset(input_jsonlines=input_jsonlines)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)

    dit = load_flow_model("flux-dev", device="cpu" if offload else device,
                          ckpt_path=f"{ckpt_path}/flux1-dev.safetensors")  # torch.bfloat16
    dit.requires_grad_(False)

    controlnet = load_controlnet(name="flux-dev", device=device)  # torch.float32
    controlnet.requires_grad_(False)
    controlnet.load_state_dict(
        torch.load(
            f'{ckpt_path}/pixelponder-fp32.bin'
        )
    )

    t5 = load_t5(device, max_length=256 if is_schnell else 512,
                 version=f"{ckpt_path}/xflux_text_encoders")  # torch.bfloat16
    t5.requires_grad_(False)
    offload_model_to_cpu(t5, offload=True)

    clip = load_clip(device, version=f"{ckpt_path}/clip-vit-large-patch14")  # torch.bfloat16
    clip.requires_grad_(False)
    ae = load_ae("flux-dev", device="cpu" if offload else device,
                 ckpt_path=f"{ckpt_path}/ae.safetensors")  # torch.float32
    ae.requires_grad_(False)

    if offload:
        offload_model_to_cpu(controlnet, dit, offload=True)

    for step, batch in enumerate(dataloader):
        img, prompt, hints, img_names = batch
        if not os.path.exists(f'{out_dir}'):
            os.makedirs(f'{out_dir}')
        cond = {}
        for k, c in hints.items():
            if len(selected_hints) > 0:
                if selected_hints[k]:
                    cond[k] = c.to(dtype=torch.float32, device=device)
            else:
                cond[k] = c.to(dtype=torch.float32, device=device)
        prompt = list(prompt)

        x = get_noise(
            img.shape[0], 512, 512, device=torch.device(device),
            dtype=torch.bfloat16, seed=seed
        )
        timesteps = get_schedule(
            num_steps=num_steps,
            image_seq_len=(512 // 8) * (512 // 8) // (16 * 16),
            shift=True,
        )
        torch.manual_seed(seed)

        if offload:
            t5, clip = t5.to(device), clip.to(device)

        inp_cond = prepare(t5=t5, clip=clip, img=x, prompt=prompt)

        if offload:
            offload_model_to_cpu(t5, clip, offload=offload)

        img = inp_cond['img'].to(dtype=torch.float32, device=device)
        txt = inp_cond['txt'].to(dtype=torch.float32, device=device)
        img_ids = inp_cond['img_ids'].to(dtype=torch.float32, device=device)
        txt_ids = inp_cond['txt_ids'].to(dtype=torch.float32, device=device)
        vec = inp_cond['vec'].to(dtype=torch.float32, device=device)
        guidance_vec = torch.full((img.shape[0],), 4.0, device=img.device, dtype=img.dtype)

        i = 0

        for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:]), desc="Denoising", total=len(timesteps[:-1])):
            if offload:
                controlnet.to(device)
            img = img.to(dtype=torch.float32, device=device)
            txt = txt.to(dtype=torch.float32, device=device)
            img_ids = img_ids.to(dtype=torch.float32, device=device)
            txt_ids = txt_ids.to(dtype=torch.float32, device=device)
            vec = vec.to(dtype=torch.float32, device=device)
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            block_res_samples = controlnet(
                img=img,
                img_ids=img_ids,
                controlnet_cond=cond,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec.to(dtype=torch.float32, device=device),
            )
            if offload:
                offload_model_to_cpu(controlnet, offload=offload)
                dit.to(device)

            img = img.to(dtype=torch.bfloat16, device=device)
            txt = txt.to(dtype=torch.bfloat16, device=device)
            img_ids = img_ids.to(dtype=torch.bfloat16, device=device)
            txt_ids = txt_ids.to(dtype=torch.bfloat16, device=device)
            vec = vec.to(dtype=torch.bfloat16, device=device)
            pred = dit(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                block_controlnet_hidden_states=[
                    0.8 * sample.to(dtype=torch.bfloat16, device=device) for sample in block_res_samples
                ],
                y=vec,
                timesteps=t_vec.to(dtype=torch.bfloat16, device=device),
                guidance=guidance_vec.to(dtype=torch.bfloat16, device=device),
            )
            if offload:
                offload_model_to_cpu(dit, offload=offload)
            img = img + (t_prev - t_curr) * pred
            i += 1
        x = img

        if offload:
            offload_model_to_cpu(dit, controlnet, offload=offload)
            ae.decoder.to(x.device)
        x = unpack(x.float(), 512, 512)
        x = ae.decode(x)
        offload_model_to_cpu(ae.decoder, offload=offload)

        x_list = [x_o.squeeze(0) for x_o in x]
        x_list = [rearrange(x1.clamp(-1, 1), "c h w -> h w c") for x1 in x_list]
        image_list = [Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy()) for x1 in x_list]

        for image, img_name in zip(image_list, img_names):
            image.save(f'{out_dir}/{img_name}')
            print(f'success save {out_dir}/{img_name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-canny', '--canny', action='store_true')
    parser.add_argument('-depth', '--depth', action='store_true')
    parser.add_argument('-hed', '--hed', action='store_true')
    parser.add_argument('-openpose', '--openpose', action='store_true')
    parser.add_argument('-batch-size', '--batch', type=int, default=1)
    parser.add_argument('-is-schnell', '--schnell', action='store_true')
    parser.add_argument('-offload', '--offload', action='store_true')
    parser.add_argument('-seed', '--seed', type=int, default=100)
    parser.add_argument('-num-steps', '--step', type=int, default=25)
    parser.add_argument('-num-gpus', '--gpu', type=int, default=2)
    parser.add_argument('-ckpt-path', '--ckptpath', type=str, default='./ckpts')
    parser.add_argument('-data-path', '--datapath', type=str, required=True)
    parser.add_argument('-save-path', '--savepath', type=str, required=True)
    args = parser.parse_args()

    # inf param init\
    selected_hints = {}
    selected_hints['canny'] = args.canny
    selected_hints['hed'] = args.hed
    selected_hints['depth'] = args.depth
    selected_hints['openpose'] = args.openpose
    save_dir_name = f'{"_canny" if selected_hints["canny"] else ""}{"_hed" if selected_hints["hed"] else ""}{"_depth" if selected_hints["depth"] else ""}{"_openpose" if selected_hints["openpose"] else ""}'
    save_dir_path = f'{args.save_path}/{save_dir_name}'

    ckpt_path = args.ckpt_path
    bs = args.batch_size
    is_schnell = args.is_schnell
    device = 'cuda:'
    offload = args.offload
    seed = args.seed
    num_steps = args.num_steps
    data_path = args.data_path
    json_path = f'{data_path}/data.jsonl'

    input_jsonlines = []
    with open(json_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            input_jsonlines.append(data)

    # mutil process param init
    gpu_nums = args.num_gpus
    available_gpu = torch.cuda.device_count()
    max_gpus = min(gpu_nums, available_gpu)

    chunk_size = len(input_jsonlines) // gpu_nums + 1
    file_chunks = [input_jsonlines[i:i + chunk_size] for i in range(0, len(input_jsonlines), chunk_size)]

    # mutil process
    multiprocessing.set_start_method('spawn', force=True)
    threads = []
    for idx in range(max_gpus):
        thread = multiprocessing.Process(target=inf,
                                         args=(
                                             bs, is_schnell, device + str(idx), offload, seed, num_steps, ckpt_path,
                                             file_chunks[idx], save_dir_path, selected_hints))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    with torch.no_grad():
        main()
