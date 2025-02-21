from pipeline import PixelPonder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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
    example_1_text = 'watercolor kitten'
    conditions_1 = {'canny': Image.open('./examples/example1/canny.png'),
                    'hed': Image.open('./examples/example1/hed.png'),
                    'depth': Image.open('./examples/example1/depth.png')}
    image1 = pixelponder(text=example_1_text, conditions=conditions_1)
    image1.save('./example1.jpg')

    plt.imshow(np.array(image1))
    plt.axis('off')
    plt.show()

    # example2
    example_2_text = 'grosse moustache acteur fracais photo exempl styles moustaches barbe hipster'
    conditions_2 = {'canny': Image.open('./examples/example2/canny.png'),
                    'hed': Image.open('./examples/example2/hed.png'),
                    'depth': Image.open('./examples/example2/depth.png'),
                    'openpose': Image.open('./examples/example2/pose.png')}

    image2 = pixelponder(text=example_2_text, conditions=conditions_2)
    image2.save('./example2.jpg')

    plt.imshow(np.array(image2))
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()