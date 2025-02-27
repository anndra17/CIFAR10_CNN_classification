# MobileNetV2 CIFAR-10 Classification with C# MVVM Integration

## Overview
This project implements image classification using **MobileNetV2** trained on the **CIFAR-10** dataset. The trained model is exported to **ONNX format** and integrated into a C# **MVVM** application for real-time image classification.

## Model Details
- **Architecture**: MobileNetV2
- **Dataset**: CIFAR-10 (10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)
- **Optimizer**: Adam with L2 weight decay (`0.001`)
- **Loss Function**: CrossEntropyLoss
- **Learning Rate Scheduler**: ReduceLROnPlateau (reduces LR by factor of 0.1 if validation loss stagnates for 5 epochs)
- **Regularization Techniques**:
  - Data Augmentation
  - L2 Weight Decay (`0.001`)
  - Learning Rate Scheduling
- **Performance**:
  - Achieves competitive accuracy on CIFAR-10
  - Tracks training and validation loss over epochs

## C# Application
The C# application is built using **MVVM** and integrates the ONNX model for real-time image classification.

### **Technologies Used**
- **.NET (C#)** for UI and model inference
- **Microsoft.ML.OnnxRuntime** for ONNX model execution
- **XAML (WPF)** for UI design
- **MVVM (Model-View-ViewModel)** design pattern

### **Structure**
#### **1. Model (CnnModel.cs)**
- Loads the ONNX model using `InferenceSession`
- Accepts input image tensors (3x32x32)
- Returns the predicted class label

#### **2. ViewModel (MainViewModel.cs)**
- Handles user interactions (Load Image, Run CNN)
- Converts images to the required format
- Passes image tensors to the ONNX model for inference
- Updates UI with the predicted class

#### **3. View (MainWindow.xaml)**
- UI elements:
  - **Button**: Load Image
  - **Image Display**: Shows the loaded image
  - **Button**: Run CNN (Performs classification)
  - **TextBlock**: Displays the predicted class

## How to Run
1. Train the MobileNetV2 model and export it as `mobilenetv2_cifar10.onnx`.
2. Place the ONNX model in the application directory.
3. Run the C# application and load an image.
4. Click "Run CNN" to classify the image.

## Future Improvements
- Improve accuracy with fine-tuning
- Expand to larger datasets
- Add real-time camera-based classification

This project demonstrates **deep learning model deployment in a C# MVVM application**, making it a practical approach for integrating machine learning into desktop applications.

