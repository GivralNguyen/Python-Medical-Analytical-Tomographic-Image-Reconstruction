import numpy as np
from skimage.transform import warp
from skimage._shared.utils import convert_to_float
from warnings import warn
import matplotlib.pyplot as plt


# Padding Function
def padding_radon(image, circle=True, *, preserve_range=False):
    """
    Pad the input image for use in the Radon transform.

    Parameters:
    - image (2D array-like): The input image for the Radon transform.
    - circle (bool, optional): If True, pads the image to form a square by cropping it to the largest inscribed circle.
                              If False, pads the image to a square using the diagonal of the image.
    - preserve_range (bool, optional): If True, the input image is converted to float while preserving its original range.
                                       If False, the input image is converted to float and normalized to the range [0, 1].

    Returns:
    - center (int): The center index of the padded image.
    - padded_image (2D array): The padded image suitable for the Radon transform.

    Raises:
    - ValueError: If the input image is not 2D or if the padded_image is not a square.

    Notes:
    - If circle is True, the input image must be zero outside the reconstruction circle. A warning is issued if this condition is not met.

    """
    if image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    
    image = convert_to_float(image, preserve_range)

    if circle:
        shape_min = min(image.shape)
        radius = shape_min // 2
        img_shape = np.array(image.shape)
        coords = np.array(np.ogrid[:image.shape[0], :image.shape[1]],
                          dtype=object)
        dist = ((coords - img_shape // 2) ** 2).sum(0)
        outside_reconstruction_circle = dist > radius ** 2
        if np.any(image[outside_reconstruction_circle]):
            warn('Radon transform: image must be zero outside the '
                 'reconstruction circle')
        # Crop image to make it square
        slices = tuple(slice(int(np.ceil(excess / 2)),
                             int(np.ceil(excess / 2) + shape_min))
                       if excess > 0 else slice(None)
                       for excess in (img_shape - shape_min))
        padded_image = image[slices]
    else:
        diagonal = np.sqrt(2) * max(image.shape)
        pad = [int(np.ceil(diagonal - s)) for s in image.shape]
        new_center = [(s + p) // 2 for s, p in zip(image.shape, pad)]
        old_center = [s // 2 for s in image.shape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
        padded_image = np.pad(image, pad_width, mode='constant',
                              constant_values=0)

    # padded_image is always square
    if padded_image.shape[0] != padded_image.shape[1]:
        raise ValueError('padded_image must be a square')
    center = padded_image.shape[0] // 2
    
     
    return center, padded_image

def radon(center, padded_image, theta=None):
    """
    Compute the Radon transform of the given padded image for the specified projection angles.

    Parameters:
    - center (int): The center index of the padded image.
    - padded_image (2D array): The padded image suitable for the Radon transform.
    - theta (array-like, optional): The projection angles in degrees. If None, a range of 180 angles from 0 to 179 degrees is used.

    Returns:
    - radon_image (2D array): The Radon transform of the padded image for the specified projection angles.

    Notes:
    - The Radon transform is a technique used in medical imaging to represent the linear projection integrals of an object.

    """
    if theta is None:
        theta = np.arange(180)
    radon_image = np.zeros((padded_image.shape[0], len(theta)),
                           dtype=padded_image.dtype)
    """
    Converts the angle from degrees to radians.
    Computes the cosine (cos_a) and sine (sin_a) of the angle.
    Constructs a 3x3 rotation matrix (R) based on the computed cosine and sine values.
    Applies the rotation matrix to the padded image using the warp function (which is not defined in the provided code snippet).
    Sum the rotated image along the vertical axis and store the result in the corresponding column of the radon_image array.
    """
    for i, angle in enumerate(np.deg2rad(theta)):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                      [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                      [0, 0, 1]])
        rotated = warp(padded_image, R, clip=False)
        radon_image[:, i] = rotated.sum(0)
    
    return radon_image

