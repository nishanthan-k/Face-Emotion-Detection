# Face-Emotion-Detection

This repository contains code and resources for a Face Emotion Detection project using TensorFlow and Haar Cascade. The project aims to detect facial expressions and classify them into different emotions, such as happy, sad, angry, etc.

## Introduction

Face Emotion Detection is an application of computer vision and deep learning that analyzes facial expressions to recognize different emotions. In this project, we utilize TensorFlow, a popular deep learning framework, along with the Haar Cascade algorithm for face detection.

## Dataset

To train the emotion detection model, we use a labeled dataset of facial expressions. Unfortunately, due to licensing restrictions, we cannot include the dataset in this repository. However, there are several publicly available datasets you can use, such as the [FER-2013](https://www.kaggle.com/msambare/fer2013) dataset from Kaggle. Ensure that you have a labeled dataset before proceeding with model training.

## Model Training

The `model_training.ipynb` notebook provides a step-by-step guide to train the emotion detection model using TensorFlow. It covers data preprocessing, model architecture, training, and evaluation. Make sure to customize the notebook according to your dataset and requirements.

## Haar Cascade

Haar Cascade is a machine learning-based object detection algorithm used for face detection. The `haarcascade_frontalface_default.xml` file in the repository contains a pre-trained Haar Cascade model for face detection. It is utilized by the `emotion_detection.py` script to detect faces in real-time.


## License

This project is licensed under the [MIT License](LICENSE).
