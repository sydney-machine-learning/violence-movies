# CNN Model for Image Classification

This repository contains a Convolutional Neural Network (CNN) implementation for image classification using TensorFlow and Keras. The model is trained on video data and can be used for binary classification tasks.

## Project Structure

- `cnn.ipynb`: Jupyter Notebook containing the CNN model implementation, training, and evaluation.

## Requirements

To run the code in this repository, you need to have the following libraries installed:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- Jupyter

You can install the required packages using the following command:

```bash
pip install tensorflow keras numpy matplotlib opencv-python-headless jupyter
```

Dataset
The dataset used for this project should be structured in the following format:
```
dataset/
    train/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
    test/
        class1/
            img1.jpg
            img2.jpg
            ...
        class2/
            img1.jpg
            img2.jpg
            ...
``` 
  
git clone https://github.com/yourusername/cnn-image-classification.git
cd cnn-image-classification

jupyter notebook cnn.ipynb
