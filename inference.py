from pipeline import PixelPonder
from PIL import Image

def main():
    controlnet_ckpt_path = "./ckpts/pixelponder-fp32.bin"
    dit_ckpt_path = './ckpts/flux1-dev.safetensors'
    t5_ckpt_path = './ckpts/xflux_text_encoders'
    clip_ckpt_path = './ckpts/clip-vit-large-patch14'
    ae_ckpt_path = './ckpts/ae.safetensors'

    pixelponder = PixelPonder(controlnet_ckpt_path=controlnet_ckpt_path,
                              dit_ckpt_path=dit_ckpt_path,
                              t5_ckpt_path=t5_ckpt_path,
                              clip_ckpt_path=clip_ckpt_path,
                              ae_ckpt_path=ae_ckpt_path,
                              offload=True)
    # example1
    example_1_text = '8_red_fox_snow_portrait'
    conditions_1 = {'canny': Image.open('./examples/example1/canny.png'),
                    'hed': Image.open('./examples/example1/hed.png'),
                    'depth': Image.open('./examples/example1/depth.png'),
                    'openpose': Image.open('./examples/example1/pose.png')}
    image1 = pixelponder(text=example_1_text, conditions=conditions_1)
    image1.save('./example1.png')

    # example2
    example_2_text = 'Set against the backdrop of a quaint, sun-drenched garden during a spring morning,\
                      it stands elegantly on a flower-strewn table, surrounded by fluttering butterflies \
                      and bees, each beam of sunlight highlighting its fanciful wings.'
    conditions_2 = {'canny': Image.open('./examples/example2/canny.png'),
                    'hed': Image.open('./examples/example2/hed.png'),
                    'depth': Image.open('./examples/example2/depth.png'),
                    'openpose': Image.open('./examples/example2/pose.png')}

    image2 = pixelponder(text=example_2_text, conditions=conditions_2)
    image2.save('./example2.png')

    # example3
    from src.flux.util import Annotator
    annotator_ckpt_path = './src/annotator/ckpts'
    # annotator_ckpt_path = None  # You can set annotator_ckpt_path to None to download the default
    condition_types = ["hed", "canny", "depth", "openpose"]

    annotator = {condition_type: Annotator(name=condition_type, device='cuda', local_dir=annotator_ckpt_path)
                 for condition_type in condition_types}
    example3_image = Image.open('image.png')
    conditions_3 = {condition_type: annotator[condition_type](
        image=example3_image, width=example3_image.size[0], height=example3_image.size[1]
    ) for condition_type in condition_types}
    example_3_text = "Stretched out along the edge of a tranquil beach, it mirrors the pastel hues of an early \
                      sunrise, waves whispering softly in the distance."
    image3 = pixelponder(text=example_3_text, conditions=conditions_3)
    image3.save('./example3.png')


if __name__ == '__main__':
    main()