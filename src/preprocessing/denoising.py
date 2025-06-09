import cv2
import numpy as np
import os

def apply_denoising(image_path):
    """
    Denoise the input image using Non-Local Means Denoising.
    
    Parameters:
    - image_path: str, path to the input image.
    
    Returns:
    - denoised_image: numpy array, the denoised image.
    """
    image = cv2.imread(image_path)
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised_image

def denoise_images_in_directory(directory_path):
    """
    Denoise all images in the specified directory.
    
    Parameters:
    - directory_path: str, path to the directory containing images.
    
    Returns:
    - denoised_images: list of numpy arrays, the denoised images.
    """
    denoised_images = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(directory_path, filename)
            denoised_image = apply_denoising(image_path)
            denoised_images.append(denoised_image)
    return denoised_images
