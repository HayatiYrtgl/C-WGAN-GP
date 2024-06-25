![alt text](https://github.com/HayatiYrtgl/C-WGAN-GP/RESULTS/100_generated.png?raw=true)
```markdown
# Conditional WGAN-GP for Image-to-Image Translation

This project implements a Conditional Wasserstein Generative Adversarial Network with Gradient Penalty (CWGAN-GP) for image-to-image translation tasks. The architecture includes both generator and discriminator models trained on a dataset of portraits.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Dataset Preparation](#dataset-preparation)
  - [Training](#training)
  - [Generating Images](#generating-images)
- [Files](#files)
- [Acknowledgements](#acknowledgements)

## Introduction

This implementation uses a CWGAN-GP to perform image-to-image translation, which can be useful for tasks such as style transfer, image inpainting, and super-resolution. The project leverages TensorFlow and Keras for building and training the models.

## Requirements

- Python 3.7+
- TensorFlow 2.0+
- Keras
- Matplotlib
- NumPy

Install the required packages using pip:

```bash
pip install tensorflow keras matplotlib numpy
```

## Usage

### Dataset Preparation

Prepare your dataset and place the images in the `../../DATASETS/portraits_dataset/` directory. Each image should be a combination of two sub-images side by side (original and transformed).

### Training

To start training the model, run:

```bash
python model.py
```

The model will train for 300 epochs, saving the generator and discriminator models every 50 epochs.

### Generating Images

During training, generated images will be saved periodically to the `created_images_cgan` directory. These images include the original, expected transformation, and the generated output.

## Files

- `dataset_creation.py`: Contains the code for preprocessing the images and creating the dataset pipeline.
- `generate_image.py`: Contains the function to generate and save images during training.
- `losses.py`: Defines the loss functions for the generator and discriminator.
- `model.py`: Main script to define and train the CWGAN-GP model, including the training loop and gradient penalty computation.

### Detailed File Descriptions

#### `dataset_creation.py`

- Preprocesses the images by reading, decoding, and splitting them into scratch and transformed parts.
- Normalizes the images and performs random horizontal flipping.
- Creates a TensorFlow dataset for training.

#### `generate_image.py`

- Generates images during training and saves them for visual inspection.
- Plots the original, expected transformation, and generated images.

#### `losses.py`

- Implements the loss functions for the generator and discriminator.
- Includes L1 loss for the generator to improve the quality of the generated images.
- Defines optimizers for both the generator and discriminator.

#### `model.py`

- Defines the CWGAN-GP model class.
- Includes the gradient penalty computation to enforce the Lipschitz constraint.
- Implements the training step for both the generator and discriminator.
- Sets up the training loop and saves models periodically using the `Monitor` callback.

## Acknowledgements

This project is based on the principles of GANs introduced by Goodfellow et al. and the improvements for stable training provided by the WGAN-GP method by Gulrajani et al. Special thanks to the open-source community for providing tools and libraries that make this implementation possible.
```

Feel free to adjust any sections or details as necessary to better fit your project's specifics and goals.
