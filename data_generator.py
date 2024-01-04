import imageio
import numpy as np
from skimage.transform import rescale, resize

class DataGenerator:

    # Generate disk knowing size, radius, center
    @staticmethod
    def generate_disk(size = (120, 120), radius = 20, center = (100, 60)):
        x, y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
        return ((x - center[0])**2 + (y - center[1])**2) < radius**2

    # Load Shepp Logan Phantom
    @staticmethod
    def load_shepp_logan_phantom():
        phantom =  imageio.imread('data/shepp_logan_phantom.png', mode='L')
        phantom = rescale(phantom, scale=0.64, mode='reflect', channel_axis=None)
        return phantom

    # Load Shepp Logan Phantom
    @staticmethod
    def load_image(path):
        phantom =  imageio.imread(path, mode='L')
        phantom = resize(phantom, (256, 256), mode='reflect', anti_aliasing=True)
        return phantom