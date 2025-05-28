Dataset(https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification?select=images)

> Total Images: 10,015
> Image Format: JPEG
> Image Size: Typically 600×450 pixels
> Segmentation Masks: Provided for each image, facilitating lesion boundary detection tasks
> Annotations: Each image is labeled with one of seven diagnostic categories:

  1. Actinic Keratoses (akiec)
  2. Basal Cell Carcinoma (bcc)
  3. Benign Keratosis-like Lesions (bkl)
  4. Dermatofibroma (df)
  5. Melanocytic Nevi (nv)
  6. Melanoma (mel)
  7. Vascular Lesions (vasc)([arXiv][1], [arXiv][2])

- Class Distribution
The dataset exhibits a significant class imbalance:

> Melanocytic Nevi : Approximately 67% of the dataset
> Melanoma : Around 11%
> Other Classes: Each constituting less than 10%

This imbalance poses challenges for training models that perform well across all classes.

- Applications

> Classification: Training models to categorize images into the seven diagnostic classes.
> Segmentation: Utilizing provided masks to train models that delineate lesion boundaries.
> Data Augmentation: Generating synthetic images to address class imbalance.
> Transfer Learning: Fine-tuning pre-trained models on this dataset for improved performance.([arXiv][2])
