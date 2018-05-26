import numpy as np
from gym.spaces.box import Box
from gym.core import Wrapper
class FrameBuffer(Wrapper):
    
    def __init__(self, env, n_frames=4):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(FrameBuffer, self).__init__(env)
        height, width = env.observation_space.shape
        obs_shape = [height, width, n_frames]
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.framebuffer = np.zeros(obs_shape, 'float32')
        
    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer
    
    def step(self, action):
        """plays breakout for 1 step, returns frame buffer"""
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(new_img)
        return self.framebuffer, reward, done, info
    
    def update_buffer(self, img):
        axis = -1
        cropped_framebuffer = self.framebuffer[:,:,:-1]
        reshaped_img = img.reshape((img.shape[0], img.shape[1], 1))
        self.framebuffer = np.concatenate([reshaped_img , cropped_framebuffer], axis = axis)
