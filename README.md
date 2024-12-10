# Animal Image Classification Project

## Overview

This project focuses on building a basic image classification model for recognizing three types of animals: **Cats**, **Dogs**, and **Foxes**. The dataset used in this project is a subset of the **Animal Image Dataset** sourced from Kaggle, containing images of these three animals. The objective of this project is to preprocess the images, perform transformations, and build a convolutional neural network (CNN) to classify the animals.

### Dataset
- **Dataset Name:** [Animal Image Dataset](https://www.kaggle.com/datasets/snmahsa/animal-image-dataset-cats-dogs-and-foxes)
- **Number of Categories:** 3 (Cats, Dogs, Foxes)
- **Total Number of Images:** 300 images (100 images per animal category)
- **Subset Used in the Project:** 15 images per category (total of 45 images)
- **Image Dimensions:** 512x512 pixels (after resizing)

### Project Objective

1. **Data Preprocessing**:
   - Resizing and padding images to 512x512 pixels.
   - Applying a pencil sketch effect to each image.
   - Creating GIFs that show the transformation from the original image to the sketched version.
   
2. **Model Development**:
   - Building a convolutional neural network (CNN) for classifying the images into one of the three animal categories (Cats, Dogs, or Foxes).

3. **Visualizing Performance**:
   - Tracking and visualizing training and validation loss curves during the model training to ensure the model is learning effectively and not overfitting.

4. **Future Improvements**:
   - Enhancing the model with pre-trained architectures and hyperparameter tuning.
   - Expanding the dataset and using data augmentation techniques to improve model generalization.

## Project Structure

The project is organized into the following directory structure:

```
Animal_Image_Classification/
│
├── data/
│   ├── cat/
│   │   ├── original/         # 15 original cat images
│   │   ├── sketched/         # 15 sketched cat images
│   │   └── gifs/             # 15 cat transformation GIFs
│   │
│   ├── dog/
│   │   ├── original/         # 15 original dog images
│   │   ├── sketched/         # 15 sketched dog images
│   │   └── gifs/             # 15 dog transformation GIFs
│   │
│   └── fox/
│       ├── original/         # 15 original fox images
│       ├── sketched/         # 15 sketched fox images
│       └── gifs/             # 15 fox transformation GIFs
│
├── model/
│   ├── cnn_model.py          # CNN model architecture definition
│   ├── train.py              # Training script
│   └── evaluate.py           # Script for model evaluation
│
├── notebooks/
│   ├── 1_data_exploration.ipynb  # Data exploration and preprocessing notebook
│   ├── 2_model_training.ipynb    # Model training and evaluation notebook
│
└── README.md                 # Project documentation
```

### Files and Directories

- **data/**: Contains the preprocessed images, including original images, sketched images, and GIFs showing the transformation.
- **model/**:
  - `cnn_model.py`: Contains the architecture of the convolutional neural network (CNN).
  - `train.py`: A script to train the CNN on the dataset.
  - `evaluate.py`: A script to evaluate the trained model and compute metrics like accuracy and loss.
- **notebooks/**:
  - `1_data_exploration.ipynb`: A Jupyter notebook for data exploration, including image transformations.
  - `2_model_training.ipynb`: A Jupyter notebook for training and evaluating the CNN model.
- **README.md**: This file provides an overview of the project and its components.

## Setup Instructions

### Prerequisites

To run this project, ensure that you have the following libraries installed:
- Python 3.6+
- TensorFlow (for model development and training)
- OpenCV (for image processing and sketching)
- Matplotlib (for visualizing loss curves)
- Pillow (for image manipulation)
- Numpy (for array manipulations)
- Keras (if you prefer using it alongside TensorFlow)

You can install the required libraries using `pip`:

```bash
pip install tensorflow opencv-python matplotlib pillow numpy
```

### Dataset Setup

1. Download the **Animal Image Dataset** from Kaggle using the following [link](https://www.kaggle.com/datasets/snmahsa/animal-image-dataset-cats-dogs-and-foxes).
2. Extract the dataset and place the images in the appropriate directories under the `data/` folder.

### Running the Project

1. **Data Preprocessing**: Run the `1_data_exploration.ipynb` notebook to explore and preprocess the dataset. This notebook will:
   - Resize and pad images to 512x512 pixels.
   - Apply a pencil sketch effect to the images.
   - Create GIFs that show the transition from the original image to the sketched version.

2. **Model Training**: Run the `2_model_training.ipynb` notebook to train the CNN on the preprocessed dataset. This notebook will:
   - Define the CNN architecture in `cnn_model.py`.
   - Train the model using the training data and monitor the loss curves.
   - Plot training and validation loss curves for visualization.
   
3. **Evaluation**: After training, run the evaluation script `evaluate.py` to assess the model's performance on the test set. The evaluation will output metrics such as accuracy, loss, precision, recall, and F1-score.

## Model Architecture

The model used in this project is a simple Convolutional Neural Network (CNN) with the following layers:
1. **Input Layer**: 512x512 RGB images.
2. **Convolutional Layers**: Multiple convolutional layers with ReLU activation and max-pooling.
3. **Flatten Layer**: Flatten the output from convolutional layers to a 1D vector.
4. **Fully Connected Layers**: Dense layers with ReLU activation.
5. **Output Layer**: 3 neurons for classifying the images as cats, dogs, or foxes, with softmax activation.

### Training and Validation Loss Curves

During training, the following loss curves will be tracked:
- **Training Loss**: Represents the loss on the training data over each epoch.
- **Validation Loss**: Represents the loss on the validation data over each epoch.

These curves will help visualize how well the model is learning and whether it is overfitting.

## Future Improvements

1. **Model Enhancements**:
   - Experiment with pre-trained models (e.g., **ResNet**, **VGG**, **EfficientNet**) using transfer learning to improve accuracy.
   - Add more convolutional layers and use batch normalization or dropout to prevent overfitting.

2. **Data Augmentation**:
   - Apply data augmentation techniques like rotation, flipping, zooming, and brightness adjustment to artificially increase the size of the dataset and improve model generalization.

3. **Larger Dataset**:
   - Use the full 300 images or more images to improve the model’s robustness and accuracy.

4. **Hyperparameter Tuning**:
   - Tune hyperparameters like learning rate, batch size, and the number of epochs to optimize model performance.

5. **Cross-Validation**:
   - Implement **k-fold cross-validation** to get more reliable performance metrics.

6. **Model Deployment**:
   - Once the model is trained and performs well, deploy it as a web or mobile application for real-time animal classification.

## Conclusion

This project provides a foundational approach to image classification using a small dataset of animal images. The dataset was preprocessed to create artistic sketch versions and GIFs for creative purposes. The model architecture uses a basic CNN, and loss curves were tracked to monitor training progress. Future improvements can be made by expanding the dataset, fine-tuning the model, and employing more advanced architectures for better performance.
