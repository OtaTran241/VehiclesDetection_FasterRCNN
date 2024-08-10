# Vehicle Detection using Faster R-CNN with PyTorch

This project implements a vehicle detection model using the Faster R-CNN architecture with a ResNet-50 backbone. The model is trained to detect and classify vehicles in images, providing bounding boxes around detected vehicles.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup and Requirements](#setup-and-requirements)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)

## Project Overview
The goal of this project is to build a vehicle detection model that can identify vehicles in images. The model is based on the Faster R-CNN architecture and uses a ResNet-50 backbone with Feature Pyramid Networks (FPN). The training and testing pipeline is built using PyTorch.

## Dataset
The dataset used in this project contains images of vehicles with corresponding COCO-style annotations

- **Train Images**: Images used for training the model.
- **Test Images**: Images used for evaluating the model.
- **COCO Annotations**: JSON files containing bounding boxes and class labels.

### Dataset Paths
- **Train Images**: `/content/Apply_Grayscale/Apply_Grayscale/Vehicles_Detection.v9i.coco/train`
- **Train Annotations**: `/content/Apply_Grayscale/Apply_Grayscale/Vehicles_Detection.v9i.coco/train/_annotations.coco.json`
- **Test Images**: `/content/Apply_Grayscale/Apply_Grayscale/Vehicles_Detection.v9i.coco/test`
- **Test Annotations**: `/content/Apply_Grayscale/Apply_Grayscale/Vehicles_Detection.v9i.coco/test/_annotations.coco.json`

## Model Architecture
The model architecture is based on Faster R-CNN with a ResNet-50 backbone:
- **Backbone**: ResNet-50 with FPN for feature extraction.
- **RPN (Region Proposal Network)**: Proposes candidate object regions.
- **ROI Heads**: Classifies and refines the proposed regions.

The model includes custom data augmentation techniques such as random horizontal flipping.

## Setup and Requirements
### Requirements
- Python 3.7+
- PyTorch
- Torchvision
- OpenCV
- Matplotlib
- Plotly

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/OtaTran241/VehiclesDetection_FasterRCNN.git
    cd vehicle-detection
    ```

## Training
The training process involves fine-tuning a pre-trained Faster R-CNN model on the vehicle detection dataset. Key hyperparameters include:
- **Batch Size**: 4
- **Learning Rate**: 0.0005
- **Epochs**: 16

To start training:
```python
pretrained_model = get_model(num_classes, pretrained=True).to(device)
train_losses = []
for epoch in range(cfg.epochs):
    epc_train_loss, epc_class_loss, epc_box_reg_loss = train_epoch(pretrained_model, train_dataloader, optimizer, lr_scheduler, epoch, logger)
    train_losses.append(epc_train_loss)
    get_report(epoch, logger, train=True)
```
## Evaluation
After training, the model is evaluated on the test dataset. The model's predictions are compared against the ground truth, and Intersection over Union (IoU) is calculated for each predicted bounding box.

Example Code:
```python
pretrained_model.eval()
for images, bbs, labels, areas, image_id in test_dataloader:
    with torch.no_grad():
        prediction = pretrained_model(images.float().to(device))

    plot_test_predictions(images[0], prediction[0]['boxes'].cpu().numpy(), 
                          prediction[0]['labels'].cpu().numpy(), 
                          prediction[0]['scores'].cpu().numpy(), 
                          cat_mapping, bbs[0].numpy(), labels[0].numpy(), image_id[0])
```
## Results
The training process shows a steady decrease in total loss over the epochs, demonstrating the model's ability to learn and improve its detection accuracy.

Loss graph over epochs:

## Usage
### Loading a Pre-trained Model
```python
loaded_model = get_model(num_classes, pretrained=True).to(device)
loaded_model.load_state_dict(torch.load('VehicleFasterRcnnModel.pth'))
```
### Making Predictions
After loading the model, you can make predictions on new images and visualize the results using the provided plot_test_predictions function.

### Saving the Model
```python
torch.save(pretrained_model.state_dict(), 'VehicleFasterRcnnModel.pth')
```
## Contributing
Contributions are welcome! If you have any ideas for improving the model or adding new features, feel free to submit a pull request or send an email to [tranducthuan220401@gmail.com](mailto:tranducthuan220401@gmail.com).
