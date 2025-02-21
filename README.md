
# PixelPonder
PixelPonder: Dynamic Patch Adaptation for Enhanced Multi-Conditional
Text-to-Image Generation


<a href=''>
  <img src='https://img.shields.io/badge/Project-Page-Green'>
</a>
<a href="">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
</a>
<a href=''>
  <img src='https://img.shields.io/badge/Paper-Arxiv-red'>
</a>
<a href="" style="margin: 0 2px;">
  <img src='https://img.shields.io/badge/Space-ZeroGPU-orange?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
</a>

# Abstract
Recent advances in diffusion-based text-to-image generation have demonstrated promising results through visual condition control. However, existing ControlNet-like methods struggle with compositional visual conditioning - simultaneously preserving semantic fidelity across multiple heterogeneous control signals while maintaining high visual quality, where they employ separate control branches that often introduce conflicting guidance during the denoising process, leading to structural distortions and artifacts in generated images. To address this issue, we present PixelPonder, a novel unified control framework, which allows for effective control of multiple visual conditions under a single control structure. Specifically, we design a patch-level adaptive condition selection mechanism that dynamically prioritizes spatially relevant control signals at the sub-region level, enabling precise local guidance without global interference. Additionally, a time-aware control injection scheme is deployed to modulate condition influence according to denoising timesteps, progressively transitioning from structural preservation to texture refinement and fully utilizing the control information from different categories to promote more harmonious image generation. Extensive experiments demonstrate that PixelPonder surpasses
previous methods across different benchmark datasets, showing superior improvement in spatial alignment accuracy while maintaining high textual semantic consistency. Qualitative evaluations confirm our method's ability to synthesize photorealistic images that faithfully adhere to complex multi-condition inputs, significantly outperforming existing compositional control approaches.


# Installation Guide
1. Clone our repo:
```bash
git clone https://github.com/
```

2. Create new virtual environment:
```bash
conda create -n pixelponder python=3.10 -y
conda activate pixelponder
```

3. Choose the appropriate version of PyTorch based on your CUDA version.: 
```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118 
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

4. Install our dependencies by running the following command:
```bash
pip install -r requirements.txt
```

# News
- **`2025/2/20`**: Inference code is released.
- **`2025/2/20`**: Our [**PixelPonder paper**](https://arxiv.org/) is available.