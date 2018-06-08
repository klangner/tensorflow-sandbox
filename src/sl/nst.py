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
import time
import argparse
import imageio
import numpy as np
import tensorflow as tf
from vgg import load_vgg_model

 
class CONFIG:
    """
    Script parameters
    """
    IMAGE_WIDTH = 400
    IMAGE_HEIGHT = 300
    COLOR_CHANNELS = 3
    NOISE_RATIO = 0.6
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
    ROOT_DIR = os.path.dirname(__file__)
    VGG_MODEL = os.path.join(ROOT_DIR, '../models/pretrained/imagenet-vgg-verydeep-19.mat')
    OUTPUT_DIR = os.path.join(ROOT_DIR, '../output/nst/')
    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]


def generate_image(model, content, style, num_iterations):
    """
    Generate image based on provided content and style images

    Arguments:
    model           -- Network model
    content         -- content image
    style           -- Style image
    num_iterations  -- number of iterations
    
    Returns
    generated image
    """    
    input_image = generate_noise_image(content)
    sess = tf.Session()
    J_content = prepare_content_cost(sess, model, content)
    J_style = prepare_style_cost(sess, model, style)
    J = total_cost(J_content, J_style, alpha=10, beta=40)

    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)
    sess.run(tf.global_variables_initializer())
    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model['input'])
        if i % 1000 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration %d:" % i)
            print("total cost = %s" % str(Jt))
            print("content cost = %s" % str(Jc))
            print("style cost = %s" % str(Js))
            
            # save current generated image in the "/output" directory
            save_image(str(i) + ".png", generated_image)
    
    return generated_image


def prepare_content_cost(sess, model, content):
    # Assign the content image to be the input of the VGG model.  
    sess.run(model['input'].assign(content))
    # Select the output tensor of layer conv4_2
    out = model['conv4_2']
    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)
    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out
    # Build the content cost graph
    J_content = compute_content_cost(a_C, a_G)
    return J_content


def prepare_style_cost(sess, model, style):
    sess.run(model['input'].assign(style))
    # Compute the style cost
    J_style = compute_style_cost(sess, model, CONFIG.STYLE_LAYERS)
    return J_style


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


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content   -- content cost
    J_style     -- style cost
    alpha       -- hyperparameter weighting the importance of the content cost
    beta        -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha*J_content + beta*J_style
    return J


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar with the cost
    """
    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    # Reshape a_C and a_G
    a_C_unrolled = tf.reshape(a_C, [-1])
    a_G_unrolled = tf.reshape(a_G, [-1])
    # compute the cost with tensorflow
    J_content = tf.reduce_sum(tf.square(a_C_unrolled-a_G_unrolled)) / (4*n_H*n_W*n_C)
    return J_content    


def compute_style_cost(sess, model, style_layers):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model        -- our tensorflow model
    style_layers -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    J_style = 0

    for layer_name, coeff in style_layers:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer
    return J_style


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.reshape(tf.transpose(a_S, [0, 3, 1, 2]), [n_C, n_H*n_W])
    a_G = tf.reshape(tf.transpose(a_G, [0, 3, 1, 2]), [n_C, n_H*n_W])
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    J_style_layer = tf.reduce_sum(tf.square(GS-GG)) / (4*((n_H*n_W)**2)*(n_C**2))
    return J_style_layer


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """    
    return tf.matmul(A, tf.transpose(A))


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Neural Style Transformer')
    parser.add_argument('content', type=str, help='content image path')
    parser.add_argument('style', type=str, help='style image path')
    parser.add_argument('-i', '--iterations', type=int, help='number of iterations', default=200)
    return parser.parse_args()
        

def main():
    args = parse_args()
    content_image = load_image(args.content)
    style_image = load_image(args.style)
    num_iterations = args.iterations if args.iterations else 200
    model = load_vgg_model(CONFIG.VGG_MODEL)
    start_time = time.time()
    generated_image = generate_image(model, content_image, style_image, num_iterations)
    total_time = time.time()-start_time
    print('Processed %d iterations in %.2f sec. (%.2f spi)' % (num_iterations, total_time, total_time/num_iterations))
    save_image('generated.jpg', generated_image)


if __name__ == "__main__":
    if not os.path.exists(CONFIG.OUTPUT_DIR):
        os.makedirs(CONFIG.OUTPUT_DIR)
    main()
