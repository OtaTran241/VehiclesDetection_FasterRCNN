# Vehicle Detection using custom ResNet backbone Faster R-CNN with PyTorch

This project implements a vehicle detection model using the Faster R-CNN architecture with a custom ResNet backbone. The model is trained to detect and classify vehicles in images, providing bounding boxes around detected vehicles.

## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)

## Dataset
The dataset used in this project contains images of vehicles with corresponding COCO-style annotations

- **Train Images**: Images used for training the model.
- **Test Images**: Images used for evaluating the model.
- **COCO Annotations**: JSON files containing bounding boxes and class labels.

### Dataset Paths
- **Train Images**: `/data/Apply_Grayscale/Apply_Grayscale/Vehicles_Detection.v9i.coco/train`
- **Train Annotations**: `/data/Apply_Grayscale/Apply_Grayscale/Vehicles_Detection.v9i.coco/train/_annotations.coco.json`
- **Test Images**: `/data/Apply_Grayscale/Apply_Grayscale/Vehicles_Detection.v9i.coco/test`
- **Test Annotations**: `/data/Apply_Grayscale/Apply_Grayscale/Vehicles_Detection.v9i.coco/test/_annotations.coco.json`

### Data augmentation
`RandomHorizontalFlip` and `ToTensor`, which are commonly used in image processing pipelines, particularly for training computer vision models.
```python
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, bboxes):
        if random.random() < self.p:
            image = F.hflip(image)
            width, _ = image.size
            if len(bboxes) != 0:
                bboxes = bboxes.clone()
                bboxes[:, [0, 2]] = width - bboxes[:, [2, 0]]
        return image, bboxes


class ToTensor(object):
    def __call__(self, image, bboxes):
        image = F.to_tensor(image)
        return image, bboxes
     

train_transform = Compose([
    RandomHorizontalFlip(p=0.5),
    ToTensor()
])

val_transform = Compose([
    ToTensor()
])

```
1. `RandomHorizontalFlip`:
- This class applies a random horizontal flip to an image, with a probability determined by p (default is 0.5).  
- The `__call__` method is invoked when an instance of the class is called with an image and its corresponding bounding boxes (bboxes).  
- If the image is flipped, the bounding boxes are also adjusted to reflect the new positions after flipping. Specifically, the x-coordinates of the bounding boxes are updated.  
  
2. `ToTensor`:
- This class converts an image to a tensor format, which is a common preprocessing step before feeding the image into a neural network.  
- The bounding boxes are left unchanged.  

3. `train_transform` and `val_transform`: 
- `train_transform` is a composition of the `RandomHorizontalFlip` and `ToTensor` transformations, used for augmenting the training data.  
- `val_transform` only applies the `ToTensor` transformation, typically used for validation data where augmentation like flipping is not desired.  

## Model Architecture
The model architecture is based on Faster R-CNN with a ResNet backbone:
- **Backbone**: A ResNet50 model pretrained on ImageNet, using layers from `conv1` to `layer3`, while excluding `layer4` to reduce complexity and improve training speed.
- **RPN (Region Proposal Network)**: Uses `AnchorGenerator` to generate anchor boxes with different sizes and aspect ratios.
- **ROI Heads**: Uses `MultiScaleRoIAlign` to extract features from region proposals.

## Setup

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/OtaTran241/VehiclesDetection_FasterRCNN.git
    cd VehiclesDetection_FasterRCNN
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
