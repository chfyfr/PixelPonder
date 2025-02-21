import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from einops import rearrange

from src.flux.modules.layers import Modulation, SelfAttention
from src.flux.math import attention
from torch import Tensor
import torch.fft as fft


class ImageStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

    def forward(self, img: Tensor, txt: Tensor, pe: Tensor, vec:Tensor) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod, _ = self.txt_mod(vec)

        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod.scale) * txt_modulated + txt_mod.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        # attn = rearrange(x, "B H L D -> B L (H D)")
        attn = attention(q, k, v, pe=pe)
        _, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        return img


class PatchSelectNetwork(nn.Module):
    def __init__(
        self, in_channel, input_shape, patch_size, type2id, select_num=None,
        depth=None, hidden_size=3072, num_heads=24, mlp_ratio=4.0, in_channels=64, qkv_bias=False
    ):
        '''
        Select Network Implementation
        Args:
            in_channel:Number of input channels, 16 for condition embed
            input_shape:Shape of the input data, (height, width)
            patch_size:Size of each patch, (patch_height, patch_width)
            type2id:Mapping from patch types to unique IDs
            select_num:represents the number of patches to be selected each time;  None means no specific depth, default=w // patch_size
            depth:Depth of the network; None means no specific depth, default=1
        '''
        super().__init__()
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.w, self.h = input_shape
        self.patch_num = (self.w//patch_size)*(self.h//patch_size)
        self.type2id = type2id
        if select_num is None:
            self.select_num = self.w//patch_size
        else:
            self.select_num = select_num
        if self.patch_num % self.select_num != 0:
            raise RuntimeError(f'patch_num:{self.patch_num} is not a multiple of select_num:{self.select_num}')

        if depth is not None:
            self.depth = depth
        else:
            self.depth = 1

        self.pos_embed_input = nn.Linear(in_channels, hidden_size, bias=True)
        self.pos_embed_output = nn.Linear(hidden_size, in_channels, bias=True)
        self.combine_attn_block = nn.ModuleList(
            [
                ImageStreamBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias
                )
                for _ in range(self.depth)
            ]
        )

        self.attn_block = nn.ModuleList(
            [
                ImageStreamBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias
                )
                for _ in range(self.depth*len(self.type2id))
            ]
        )


    def image_into_patch(self, ori_img, norm=True):
        '''
        To convert an image into patches
        '''
        if ori_img.ndim == 4:
            img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c',
                                    p1=self.patch_size, p2=self.patch_size)
            if norm:
                img_squeeze = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (
                            img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            #  p= pow(self.patch_size, 2),c= self.in_channel
            img_patch = rearrange(img_squeeze, 'b n p c -> b n (p c)')
            return img_patch
        elif ori_img.ndim == 3:
            img_squeeze = rearrange(ori_img, 'c (h p1) (w p2) -> (h w) (p1 p2) c',
                                    p1=self.patch_size, p2=self.patch_size)
            if norm:
                img_squeeze = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (
                            img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            #  p= pow(self.patch_size, 2),c= self.in_channel
            img_patch = rearrange(img_squeeze, 'n p c -> n (p c)')
            return img_patch
        raise RuntimeError(f'This operation of patching images of {ori_img.ndim} dimension is not provided.')


    def patch_into_image(self, ori_patch):
        '''
        To convert patches into image,It is the inverse operation of image_into_patch.
        '''
        patch_squeeze = rearrange(ori_patch, 'b n (p c) -> b n p c', p=pow(self.patch_size, 2))
        img = rearrange(patch_squeeze, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                        p1=self.patch_size, p2=self.patch_size, h=self.h//self.patch_size, w=self.w//self.patch_size)
        return img


    def embed_into_block(self, embed:Tensor):  #  (b,c,h,w) -> (b,n,hidden)
        embed = rearrange(embed, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2,
                                        pw=2)  # (b,n,4c)
        embed = self.pos_embed_input(embed)  # (b,n,hidden)
        return embed


    def block_into_embed(self, embed:Tensor):  #  (b,n,hidden) -> (b,c,h,w)
        embed = self.pos_embed_output(embed)
        embed = rearrange(embed, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                          ph=2, pw=2, h=self.h//2, w=self.w//2)
        return embed


    def update_mask(self, zero_mask:Tensor, indices:Tensor):
        b, n = zero_mask.shape

        zero_mask.scatter_(1, indices, False)

        update_mask = torch.zeros((b, n), dtype=torch.bool, device=indices.device)
        update_mask.scatter_(1, indices, True)
        return zero_mask, update_mask


    def combine_patchs(self, conds:dict, selected_patch:Tensor, txt:Tensor, pe:Tensor, vec:Tensor):
        '''
        the Implementation of Select Network forward
        '''
        selected_patch = self.image_into_patch(selected_patch)  # (b,n,pc)
        zero_mask = torch.all(torch.ones_like(selected_patch, dtype=torch.bool, device=txt.device), dim=-1) #(b,n)

        ori_patch_embed = torch.cat([self.image_into_patch(v, norm=False).unsqueeze(-2) for _,v in conds.items()], dim=-2)

        for i in range(self.patch_num//self.select_num):
            patchs = []
            for k, v in conds.items():  # v:(b,c,w,h)
                patch = self.embed_into_block(v)
                for depth in range(len(self.attn_block)//len(self.type2id)):
                    patch = self.attn_block[self.type2id[k] * (depth+1)](img=patch, txt=txt, pe=pe, vec=vec)
                patch = self.block_into_embed(patch)
                patchs.append(self.image_into_patch(patch).unsqueeze(-2))  # (b,c,w,h)->(b,n,pc)->concat[(b,n,1,pc)*type]=(b,n,type,pc)
            patch_embed = torch.cat(patchs, dim=-2)  # (b,n,3,pc)

            attn_comb_patch = self.patch_into_image(selected_patch.clone())
            attn_comb_patch_tmp = self.embed_into_block(attn_comb_patch)
            for layer in self.combine_attn_block:
                attn_comb_patch_tmp = layer(attn_comb_patch_tmp, txt, pe, vec)
            attn_comb_patch = self.block_into_embed(attn_comb_patch_tmp)
            attn_comb_patch = self.image_into_patch(attn_comb_patch)  #(b,n,pc)

            embed = patch_embed + attn_comb_patch.unsqueeze(-2)  # (b,n,pc)->(b,n,1,pc)->(b,n,3,pc)
            x = torch.sum(embed, dim=-1, keepdim=False)  # (b,n,3,pc)->(b,n,3)
            x = F.normalize(x, p=2, dim=-1)
            softmax_result = F.softmax(x, dim=-1)  # (b,n,3)
            softmax_result[~zero_mask] = 0.
            #  Select the patches that were chosen in this step.
            max_probs, all_indices = torch.max(softmax_result, dim=-1)  # max_probs:(b,n)
            all_indices = all_indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,ori_patch_embed.shape[-1])
            selected_patch = ori_patch_embed.gather(-2, all_indices).squeeze(-2)
            #  Select the top num patches from the chosen patches.
            _, top_n_indices = torch.topk(max_probs, self.select_num, dim=-1)  # (b, self.select_num)

            #  zero_mask:(b,n)    update_mask_patch:(b,n)     update_mask_combined_patch:(b,n)
            zero_mask, update_mask = self.update_mask(zero_mask, top_n_indices)

            selected_patch[update_mask] += selected_patch[update_mask]

            for k, v in conds.items():
                v = self.image_into_patch(v, norm=False)
                v[update_mask] = 0
                v = self.patch_into_image(v)
                conds[k] = v

        return self.patch_into_image(selected_patch)


    def forward(self, conds, txt, pe, vec):
        bs = 0

        for k,v in conds.items():
            if len(conds) == 1:
                return v
            if bs == 0:
                bs = v.size(0)
                break

        selected_patch = torch.zeros((bs, self.in_channel, self.w, self.h), dtype=txt.dtype, device=txt.device)
        selected_patch = self.combine_patchs(conds=conds, selected_patch=selected_patch, txt=txt, pe=pe, vec=vec)

        return selected_patch
