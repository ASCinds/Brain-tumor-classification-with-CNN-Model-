# Brain Tumor Classification using Convolutional Neural Networks (CoNvNet)

## Introduction

This project focuses on the development of a Convolutional Neural Network (CNN) for the classification of brain tumor MRI images. The dataset consists of 44 classes, each representing different types and subtypes of brain tumors. The CNN model is designed to automatically learn relevant features from the medical images, contributing to early detection and diagnosis.

## Project Structure

The project is organized into several modules, each serving a specific purpose:

1. 'create_dataset.py': Defines a custom dataset class for handling medical image data.

2. 'prepare_dataset.py': Prepares the dataset for training, testing, and validation, including data splitting and DataFrame construction.

3. 'model.py': Defines the CNN model (CoNvNet) architecture for brain tumor classification.

4. 'main.py': Organizes the training and evaluation of the CNN model.

5. 'metrics.py': Computes and evaluates various metrics on the model's predictions.

6. 'plots.py': Creates plots to visualize the training and validation process.

7. 'predict.py': Uses the trained model for making predictions on new images.






## Setting up Environment

### Download Requirements
1. Open Command Prompt (Windows) or Terminal (macOS/Linux).
2. Navigate to your project directory:
   
   cd path\to\the\project

3. Install dependencies from the requirements.txt file:

    pip install -r requirements.txt



## Running Scripts

1. Navigate to Project Directory:

   cd path\to\the\project\module

2. Activate Virtual Environment:
 
   .\env\Scripts\activate

3.Use the 'python' command to run your script:

         python your_script.py



## Making Predictions

1. Run Predict Script
    Open a command prompt or terminal.
    Navigate to your project directory:
    
     cd path\to\your\project
    
    -Run the predict.py script:
   
     python predict.py
     

2. Access Local Host:
   - After running the script, the command prompt will display a local host link.
   - Copy the link provided and open it in your browser.

3. Upload Images for Prediction:
   - Once the local host page is open in your browser, you'll see an interface for image uploads.
   - Click on the upload button and select the images you want to predict.
   - Submit the images, and the model will provide predictions based on the uploaded images.






