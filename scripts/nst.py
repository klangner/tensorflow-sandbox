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
#
# Custom images should have size (300, 225) (width, height)

import os
import argparse
import imageio
import numpy as np
import tensorflow as tf
from vgg import load_vgg_model


class CONFIG:
    IMAGE_WIDTH = 400
    IMAGE_HEIGHT = 300
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
    ROOT_DIR = os.path.dirname(__file__)
    VGG_MODEL = os.path.join(ROOT_DIR, '../models/pretrained/imagenet-vgg-verydeep-19.mat')
    OUTPUT_DIR = os.path.join(ROOT_DIR, '../output/nst/')


def generate_image(model, content, style, num_iterations):
    """
    Generate image based on provided content and style images
    """    
    return generate_noise_image(content)


def generate_noise_image(content_image, noise_ratio = CONFIG.NOISE_RATIO):
    """
    Generates a noisy image by adding random noise to the content_image
    """    
    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20, (1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)).astype('float32')    
    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)    
    return input_image


def load_image(path):
    """
    Reshape and normalize the input image (content or style)
    """
    image = imageio.imread(path)
    # Reshape image to mach expected input of VGG16
    image = np.reshape(image, ((1,) + image.shape))    
    # Substract the mean to match the expected input of VGG16
    image = image - CONFIG.MEANS    
    return image


def save_image(file_name, image):    
    """
    Un-normalize and save image
    """
    # Un-normalize the image so that it looks good
    image = image + CONFIG.MEANS    
    # Clip and Save the image
    image = np.clip(image[0], 0, 255).astype('uint8')
    imageio.imwrite(CONFIG.OUTPUT_DIR + file_name, image)


def main(args):
    content_image = load_image(args.content)
    style_image = load_image(args.style)
    num_iterations = args.iterations if args.iterations else 200
    model = load_vgg_model(CONFIG.VGG_MODEL)
    generated_image = generate_image(model, content_image, style_image, num_iterations)
    save_image('generated.jpg', generated_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Style Transformer')
    parser.add_argument('-c', '--content', type=str, help='content image path')
    parser.add_argument('-s', '--style', type=str, help='style image path')
    parser.add_argument('-i', '--iterations', type=int, help='number of iterations')
    main(parser.parse_args())
