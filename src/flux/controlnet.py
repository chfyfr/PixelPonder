from dataclasses import dataclass

import torch
from torch import Tensor, nn
from einops import rearrange

from .modules.layers import (DoubleStreamBlock, EmbedND, MLPEmbedder, timestep_embedding, ZeroSingleStreamBlock, ZeroDoubleStreamBlock)
from src.flux.SelectNet import PatchSelectNetwork
import torch.fft as fft

@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNetFlux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    _supports_gradient_checkpointing = True

    def __init__(self, params: FluxParams, controlnet_depth: int = 2, patch_size: int = 2, type2id=None):
        super().__init__()
        self.patch_size = patch_size
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                ZeroDoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(controlnet_depth)
            ]
        )

        total_params = sum(p.numel() for p in self.double_blocks.parameters())
        print(f"Total number of parameters: {total_params}")

        self.single_blocks = nn.ModuleList(
            [
                ZeroSingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(controlnet_depth * 2)
            ]
        )

        self.pos_embed_input = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.gradient_checkpointing = False
        if type2id is None:
            self.type2id = {"canny": 0, "openpose": 1,
                            "depth": 2, "hed": 3}
        else:
            self.type2id = type2id
        self.hint_types = len(self.type2id)
        self.input_hint_block_list_moe = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.SiLU()
        ) for _ in range(self.hint_types)])
        self.input_hint_block_share = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(nn.Conv2d(16, 16, 3, padding=1))
        )

        self.select_net = PatchSelectNetwork(
            in_channel=16,
            input_shape=(512//8, 512//8),
            patch_size=patch_size,
            type2id=self.type2id,
            select_num=None,
            depth=1
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        controlnet_cond: dict,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        vec = self.time_in(timestep_embedding(timesteps, 256)) # [1,3072]
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        img = self.img_in(img)
        conds = {}
        for k,v in controlnet_cond.items():
            id = self.type2id[k]
            v = self.input_hint_block_list_moe[id](v)
            v = self.input_hint_block_share(v)
            conds[k] = v

        cond = self.select_net(conds, txt, pe, vec)
        cond = rearrange(cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        cond = self.pos_embed_input(cond)
        img = img + cond

        controlnet_block_res_samples = ()
        for block in self.double_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                img, txt = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    txt,
                    vec,
                    pe,
                )
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            controlnet_block_res_samples = controlnet_block_res_samples + (img, )

        img = torch.cat([img, txt], dim=1)

        for block in self.single_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                img = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    vec,
                    pe,
                )
            else:
                img = block(img, vec=vec, pe=pe)
            controlnet_block_res_samples = controlnet_block_res_samples + (img, )

        return controlnet_block_res_samples
