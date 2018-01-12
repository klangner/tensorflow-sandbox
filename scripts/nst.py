#
# Neural Style Transfer
# https://arxiv.org/abs/1508.06576
#
# Parameters:
#  * --content  - Content image path
#  * --style    - Style image path 
#  * --epoch    - Number of iterations. Default 200
#
# This script also assumes that:
#  * VGG-19 model is in ../models/pretrained folder
#  * output image is written to ../output/nst/generated.jpg

import os
import argparse
import imageio
import numpy as np
import tensorflow as tf
from vgg import load_vgg_model


FILE_DIR = os.path.dirname(__file__)
VGG_MODEL = os.path.join(FILE_DIR, '../models/pretrained/imagenet-vgg-verydeep-19.mat')
OUTPUT_DIR = os.path.join(FILE_DIR, '../output/nst/')


def save_image(file_name, image):    
    # Un-normalize the image so that it looks good
    # image = image + CONFIG.MEANS    
    # Clip and Save the image
    # image = np.clip(image[0], 0, 255).astype('uint8')
    imageio.imwrite(OUTPUT_DIR + file_name, image)


def generate_image(model, content, style):
    return content


def main(args):
    content_image = imageio.imread(args.content)
    style_image = imageio.imread(args.style)
    num_iterations = args.iterations if args.iterations else 200
    model = load_vgg_model(VGG_MODEL)
    generated_image = generate_image(model, content_image, style_image)
    save_image('generated.jpg', generated_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Style Transformer')
    parser.add_argument('-c', '--content', type=str, help='content image path')
    parser.add_argument('-s', '--style', type=str, help='style image path')
    parser.add_argument('-i', '--iterations', type=int, help='number of iterations')
    main(parser.parse_args())
