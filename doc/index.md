This [repository](https://github.com/cheeweeng/Training-and-Fine-tuning-an-image-recognition-and-classification-model) contains the 
python codes to develop an image classification model utilizing MobileNetV2  and submitted in June 2026 for the Deep Learning for Image Recognition course assignment 
in Ngee Ann Polytechnic. 

# Overview  
The Deep Learning (DLIR) assignment for the Specialist Diploma in Applied Generative AI tasks students with a comprehensive computer vision project.  

### The Problem and Objective   
The core objective of this assignment is to build and evaluate deep learning image classification models capable of recognizing and categorizing ten different types of food. Accurately classifying images requires the application of neural networks to process raw image data and map it to specific food classes. The assignment specifies a structured, four-step approach to properly address this image classification problem:
1.	Data Loading and Preprocessing: 
A dataset which is pre-divided into specific subsets: a training set containing 750 images per food type, a validation set with 200 images per food type, and a testing set with 50 images per food type is provided (download from POLITEMALL). First, students must load the provided training, validation, and test image datasets into Jupyter Notebook. The images should be uniformly resized, with a recommended resolution of 150x150 pixels.
2.	Model Development: 
The core of the approach involves developing at least two distinct image classification models. The first model must be built entirely from scratch, utilizing standard 2D convolutional (conv2D) and dense layers. The second model must leverage the power of pre-trained networks. For both approaches, students are required to follow a universal machine learning workflow: establishing a baseline model, scaling it up until overfitting occurs, applying necessary regularization techniques, and rigorously tuning hyperparameters. Throughout this training phase, model performance curves must be carefully recorded for later analysis.
3.	Model Evaluation: 
After the training phase is complete, both models must be evaluated against the unseen testing dataset. Students must compare the performance of both models to determine which approach yields superior results during testing and recommend the single best model based.
4.	Real-World Prediction: 
To test the practical robustness of the recommended best model, students must source at least three novel food images directly from the internet. These external images are then fed into the selected model to verify if it can accurately classify real-world data outside of the provided dataset.

Ultimately, this structured approach forms the basis of the required final deliverables, which include a set of final presentation PowerPoint slides, a comprehensive individual report (Word document) detailing the methodology, and a Jupyter notebook (multiple notebooks acceptable) containing the functional code.
