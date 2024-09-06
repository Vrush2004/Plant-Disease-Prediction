# Plant Disease Prediction

## Overview
This project leverages a Convolutional Neural Network (CNN) to predict plant diseases from images. The project includes the entire pipeline, from dataset preparation to model training and deployment via a Streamlit app. The model is trained on the PlantVillage dataset, a widely used dataset for plant disease detection.

## Features
- **Image Classification**: Predicts plant diseases using deep learning.
- **Streamlit Web App**: User-friendly interface for uploading images and getting predictions.
- **Data Augmentation**: Utilizes `ImageDataGenerator` to augment the training data.
- **Model Training**: Includes code for training the model on the PlantVillage dataset.
- **Model Deployment**: The trained model is deployed in a Streamlit application.

## Technology Stack
- **Python**
- **TensorFlow** for deep learning
- **Keras** for model building
- **Flask** (for any additional backend requirements)
- **Streamlit** for app deployment
- **Kaggle API** for dataset download
- **PIL** (Python Imaging Library) for image processing
- **Matplotlib** for data visualization
- **NumPy** for numerical computations

## Model Architecture

- **Input Layer**: Accepts images of size 224x224x3.
- **Conv2D Layers**: Multiple convolutional layers with ReLU activation and MaxPooling.
- **Dense Layers**: Fully connected layers with ReLU activation.
- **Output Layer**: Softmax activation for multi-class classification.

### Prerequisites
- Python 3.7 or higher
- Kaggle API (for downloading the dataset)
- Virtual environment (optional but recommended)

### Run the Streamlit App

![output](https://github.com/user-attachments/assets/e8bc2801-3e65-463d-9953-50dce940068d)

To start the Streamlit application, run the following command:

```bash
streamlit run main.py

