from flask import Flask, request, jsonify, send_from_directory
import torch
from PIL import Image
import numpy as np
import os
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Model initialization
print("Loading models... This may take a moment.")

# YOLOv8 model
yolo_model = None
try:
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")  # Using the nano model for faster inference
    print("YOLOv8 model loaded successfully")
except Exception as e:
    print("Error loading YOLOv8 model:", e)
    yolo_model = None

# DETR model (DEtection TRansformer)
detr_processor = None
detr_model = None
try:
    from transformers import DetrImageProcessor, DetrForObjectDetection
    
    detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    print("DETR model loaded successfully")
except Exception as e:
    print("Error loading DETR model:", e)
    detr_processor = None
    detr_model = None

# ViT model
vit_processor = None
vit_model = None
try:
    from transformers import ViTImageProcessor, ViTForImageClassification
    vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    print("ViT model loaded successfully")
except Exception as e:
    print("Error loading ViT model:", e)
    vit_processor = None
    vit_model = None

# Get device information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def image_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def process_yolo(image):
    if yolo_model is None:
        return {"error": "YOLOv8 model not loaded"}
    
    # Measure inference time
    start_time = time.time()
    
    # Convert to numpy if it's a PIL image
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
        
    # Run inference
    results = yolo_model(image_np)
    
    # Process results
    result_image = results[0].plot()
    result_image = Image.fromarray(result_image)
    
    # Get detection information
    boxes = results[0].boxes
    class_names = results[0].names
    
    # Format detection results
    detections = []
    for box in boxes:
        class_id = int(box.cls[0].item())
        class_name = class_names[class_id]
        confidence = round(box.conf[0].item(), 2)
        bbox = box.xyxy[0].tolist()
        bbox = [round(x) for x in bbox]
        detections.append({
            "class": class_name,
            "confidence": confidence,
            "bbox": bbox
        })
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    # Add inference time and device info
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    
    return {
        "image": image_to_base64(result_image),
        "detections": detections,
        "performance": {
            "inference_time": round(inference_time, 3),
            "device": device_info
        }
    }

def process_detr(image):
    if detr_model is None or detr_processor is None:
        return {"error": "DETR model not loaded"}
    
    # Measure inference time
    start_time = time.time()
    
    # Prepare image for the model
    inputs = detr_processor(images=image, return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = detr_model(**inputs)
    
    # Process results
    target_sizes = torch.tensor([image.size[::-1]])
    results = detr_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]
    
    # Create a copy of the image to draw on
    result_image = image.copy()
    fig, ax = plt.subplots(1)
    ax.imshow(result_image)
    
    # Format detection results
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i) for i in box.tolist()]
        class_name = detr_model.config.id2label[label.item()]
        confidence = round(score.item(), 2)
        
        # Draw rectangle
        rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                         linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        plt.text(box[0], box[1], "{}: {}".format(class_name, confidence), 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        detections.append({
            "class": class_name,
            "confidence": confidence,
            "bbox": box
        })
    
    # Save figure to image
    buf = io.BytesIO()
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    result_image = Image.open(buf)
    plt.close(fig)
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    # Add inference time and device info
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    
    return {
        "image": image_to_base64(result_image),
        "detections": detections,
        "performance": {
            "inference_time": round(inference_time, 3),
            "device": device_info
        }
    }

def process_vit(image):
    if vit_model is None or vit_processor is None:
        return {"error": "ViT model not loaded"}
    
    # Measure inference time
    start_time = time.time()
    
    # Prepare image for the model
    inputs = vit_processor(images=image, return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = vit_model(**inputs)
        logits = outputs.logits
    
    # Get the predicted class
    predicted_class_idx = logits.argmax(-1).item()
    prediction = vit_model.config.id2label[predicted_class_idx]
    
    # Get top 5 predictions
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    top5_prob, top5_indices = torch.topk(probs, 5)
    
    results = []
    for i, (prob, idx) in enumerate(zip(top5_prob, top5_indices)):
        class_name = vit_model.config.id2label[idx.item()]
        results.append({
            "rank": i+1,
            "class": class_name,
            "probability": round(prob.item(), 3)
        })
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    # Add inference time and device info
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    
    return {
        "top_predictions": results,
        "performance": {
            "inference_time": round(inference_time, 3),
            "device": device_info
        }
    }

@app.route('/api/detect/yolo', methods=['POST'])
def yolo_detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    image = Image.open(file.stream)
    
    result = process_yolo(image)
    return jsonify(result)

@app.route('/api/detect/detr', methods=['POST'])
def detr_detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    image = Image.open(file.stream)
    
    result = process_detr(image)
    return jsonify(result)

@app.route('/api/classify/vit', methods=['POST'])
def vit_classify():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    image = Image.open(file.stream)
    
    result = process_vit(image)
    return jsonify(result)

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "online",
        "models": {
            "yolo": yolo_model is not None,
            "detr": detr_model is not None and detr_processor is not None,
            "vit": vit_model is not None and vit_processor is not None
        },
        "device": "GPU" if torch.cuda.is_available() else "CPU"
    })

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
