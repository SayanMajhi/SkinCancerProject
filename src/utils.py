def load_ground_truth(csv_file):
    import pandas as pd
    return pd.read_csv(csv_file)

def get_image_paths(data_dir):
    import os
    return [os.path.join(data_dir, 'images', f) for f in os.listdir(os.path.join(data_dir, 'images')) if f.endswith('.jpg')]

def get_mask_paths(data_dir):
    import os
    return [os.path.join(data_dir, 'masks', f) for f in os.listdir(os.path.join(data_dir, 'masks')) if f.endswith('.png')]

def preprocess_image(image):
    from skimage import io, color
    import numpy as np
    image = io.imread(image)
    image = color.rgb2gray(image)
    return np.expand_dims(image, axis=-1)

def one_hot_encode(labels):
    import numpy as np
    classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    return np.array([[1 if label == cls else 0 for cls in classes] for label in labels])