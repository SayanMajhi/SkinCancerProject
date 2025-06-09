# Deep Learning Workshop on HAM10000 Skin Cancer Dataset

This workshop focuses on building a deep learning model for skin cancer classification using the HAM10000 dataset. The project includes various stages of data processing, augmentation, model training, and evaluation.

## Project Structure

```
deeplearning-ham10000-workshop
├── data
│   ├── images          # Contains images from the HAM10000 dataset
│   ├── masks           # Contains masks for segmentation tasks
│   └── GroundTruth.csv  # CSV file with ground truth labels
├── notebooks
│   ├── 01_data_preprocessing.ipynb  # Initial data preprocessing
│   ├── 02_data_augmentation_gan.ipynb # Data augmentation using GANs
│   ├── 03_color_constancy_denoising.ipynb # Color constancy and denoising
│   ├── 04_cnn_self_attention.ipynb  # Deep CNN with self-attention
│   ├── 05_transfer_learning.ipynb    # Transfer learning with pretrained models
│   ├── 06_ensemble_learning.ipynb     # Ensemble learning techniques
│   └── 07_evaluation.ipynb            # Model evaluation
├── src
│   ├── data_augmentation
│   │   └── gan.py                     # GAN implementation for data augmentation
│   ├── preprocessing
│   │   ├── color_constancy.py         # Color constancy algorithms
│   │   └── denoising.py               # Denoising techniques
│   ├── models
│   │   ├── cnn_self_attention.py       # Deep CNN with self-attention
│   │   ├── resnet50.py                 # ResNet50 model for transfer learning
│   │   ├── efficientnet.py             # EfficientNet model for transfer learning
│   │   └── ensemble.py                 # Ensemble learning methods
│   └── utils.py                       # Utility functions
├── requirements.txt                    # Project dependencies
├── README.md                           # Project documentation
└── .gitignore                          # Files to ignore in version control
```

## Dataset

The HAM10000 dataset consists of images of skin lesions, along with corresponding masks for segmentation tasks. The dataset is structured as follows:

- **data/images**: Contains the images from the HAM10000 dataset.
- **data/masks**: Contains the masks corresponding to the images.
- **data/GroundTruth.csv**: A CSV file containing the ground truth labels with the following columns:
  - image
  - MEL
  - NV
  - BCC
  - AKIEC
  - BKL
  - DF
  - VASC

## Workshop Overview

1. **Data Preprocessing**: Load the dataset and perform basic exploration.
2. **Data Augmentation**: Enhance the dataset using Generative Adversarial Networks (GANs).
3. **Image Preprocessing**: Apply color constancy and denoising techniques to improve image quality.
4. **Model Development**: Build and train a Deep CNN classifier with a self-attention mechanism.
5. **Transfer Learning**: Utilize pretrained models like ResNet50 and EfficientNet for skin cancer classification.
6. **Ensemble Learning**: Combine predictions from multiple models for improved classification accuracy.
7. **Model Evaluation**: Evaluate the performance of the models using appropriate metrics and visualizations.

## Requirements

To run this project, you will need the following dependencies:

- TensorFlow or PyTorch
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn
- Other relevant libraries

You can install the required packages using the following command:

```
pip install -r requirements.txt
```

## Getting Started

1. Clone the repository:
   ```
   git clone (https://github.com/SayanMajhi/SkinCancerProject.git)
   ```
2. Navigate to the project directory:
   ```
   cd deeplearning-ham10000-workshop
   ```
3. Install the required dependencies.
4. Start exploring the Jupyter notebooks in the `notebooks` directory to follow along with the workshop.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
