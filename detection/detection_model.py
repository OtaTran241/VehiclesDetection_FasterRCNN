from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os
import random
import uuid

# Từ điển nhãn (class labels)
num_classes_dict = {0: 'cars', 1: 'Bus', 2: 'Car', 3: 'Motorcycle', 4: 'Pickup', 5: 'Truck'}

# Tạo một từ điển màu cho mỗi class
color_dict = {
    'cars': (255, 0, 0),      # Đỏ
    'Bus': (0, 255, 0),       # Xanh lá
    'Car': (0, 0, 255),       # Xanh dương
    'Motorcycle': (255, 255, 0), # Vàng
    'Pickup': (255, 0, 255),  # Hồng
    'Truck': (0, 255, 255)    # Cyan
}

# Hàm load model Faster R-CNN
def get_model(num_classes, pretrained=True):
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def load_model():
    num_classes = len(num_classes_dict)
    model = get_model(num_classes, pretrained=False)
    model.load_state_dict(torch.load("model/VehicleFasterRcnnModel.pth", map_location=torch.device('cpu')))
    model.eval()  # Thiết lập mô hình ở chế độ đánh giá
    return model

# Hàm dự đoán và vẽ bounding boxes
def predict_image(model, image):
    # Chuyển đổi ảnh thành tensor
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)  # Thêm batch dimension
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Vẽ bounding boxes trên ảnh mới (annotated image)
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Tăng kích thước font lên 20
    except IOError:
        font = ImageFont.load_default()
    
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if score > 0.5:  # Chỉ vẽ các box có độ tin cậy > 50%
            box = box.tolist()
            class_name = num_classes_dict[label.item()]
            color = color_dict[class_name]
            draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=3)
            
            text = f"{class_name}: {score:.2f}"
            text_position = (box[0], box[1] - 25)
            
            draw.text(text_position, text, fill=color, font=font)
    
    # Tạo tên file duy nhất cho ảnh kết quả
    unique_filename = f"annotated_{uuid.uuid4()}.jpg"
    annotated_path = os.path.join('temp', unique_filename)
    annotated_image.save(annotated_path)
    
    return annotated_path

# Hàm run_detection: xử lý file ảnh đầu vào và trả về ảnh có bounding boxes
def run_detection(file_path):
    # Load ảnh từ đường dẫn
    image = Image.open(file_path).convert("RGB")
    
    # Load mô hình
    model = load_model()
    
    # Dự đoán và vẽ bounding boxes
    output_image_path = predict_image(model, image)
    
    return {"output_image": output_image_path, "message": "Vehicle detection complete!"}
