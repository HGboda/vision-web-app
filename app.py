import gradio as gr
import torch
from PIL import Image
import numpy as np
import os

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
    
    # Load the DETR image processor
    # DetrImageProcessor: Handles preprocessing of images for DETR model
    # - Resizes images to appropriate dimensions
    # - Normalizes pixel values
    # - Converts images to tensors
    # - Handles batch processing
    detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # Load the DETR object detection model
    # DetrForObjectDetection: The actual object detection model
    # - Uses ResNet-50 as backbone
    # - Transformer-based architecture for object detection
    # - Predicts bounding boxes and object classes
    # - Pre-trained on COCO dataset by Facebook AI Research
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
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define model inference functions
def process_yolo(image):
    if yolo_model is None:
        return None, "YOLOv8 model not loaded"
    
    # Measure inference time
    import time
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
        detections.append("{}: {} at {}".format(class_name, confidence, bbox))
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    # Add inference time and device info to detection text
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    performance_info = f"\n\nInference time: {inference_time:.3f} seconds on {device_info}"
    detection_text = "\n".join(detections) if detections else "No objects detected"
    detection_text += performance_info
    
    return result_image, detection_text

def process_detr(image):
    if detr_model is None or detr_processor is None:
        return None, "DETR model not loaded"
    
    # Measure inference time
    import time
    start_time = time.time()
    
    # Prepare image for the model
    inputs = detr_processor(images=image, return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = detr_model(**inputs)
    
    # Convert outputs to image with bounding boxes
    # Create tensor with original image dimensions (height, width)
    # image.size[::-1] reverses the (width, height) to (height, width) as required by DETR
    target_sizes = torch.tensor([image.size[::-1]])
    
    # Process raw model outputs into usable detection results
    # - Maps predictions back to original image size
    # - Filters detections using confidence threshold (0.9)
    # - Returns a dictionary with 'scores', 'labels', and 'boxes' keys
    # - [0] extracts results for the first (and only) image in the batch
    results = detr_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]
    
    # Create a copy of the image to draw on
    result_image = image.copy()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import io
    
    # Create figure and axes
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
        
        detections.append("{}: {} at {}".format(class_name, confidence, box))
    
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
    
    # Add inference time and device info to detection text
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    performance_info = f"\n\nInference time: {inference_time:.3f} seconds on {device_info}"
    detection_text = "\n".join(detections) if detections else "No objects detected"
    detection_text += performance_info
    
    return result_image, detection_text

def process_vit(image):
    if vit_model is None or vit_processor is None:
        return "ViT model not loaded"
    
    # Measure inference time
    import time
    start_time = time.time()
    
    # Prepare image for the model
    inputs = vit_processor(images=image, return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = vit_model(**inputs)
        # Extract raw logits (unnormalized scores) from model output
        # Hugging Face models return logits directly, not probabilities
        logits = outputs.logits
    
    # Get the predicted class
    # argmax(-1) finds the index with highest score across the last dimension (class dimension)
    # item() converts the tensor value to a Python scalar
    predicted_class_idx = logits.argmax(-1).item()
    # Map the class index to human-readable label using the model's configuration
    prediction = vit_model.config.id2label[predicted_class_idx]
    
    # Get top 5 predictions
    # Apply softmax to convert raw logits to probabilities
    # softmax normalizes the exponentials of logits so they sum to 1.0
    # dim=-1 applies softmax along the class dimension
    # Shape before softmax: [1, num_classes] (batch_size=1, num_classes=1000)
    # [0] extracts the first (and only) item from the batch dimension
    # Shape after [0]: [num_classes] (a 1D tensor with 1000 class probabilities)
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    # Get the values and indices of the 5 highest probabilities
    top5_prob, top5_indices = torch.topk(probs, 5)
    
    results = []
    for i, (prob, idx) in enumerate(zip(top5_prob, top5_indices)):
        class_name = vit_model.config.id2label[idx.item()]
        results.append("{}. {}: {:.3f}".format(i+1, class_name, prob.item()))
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    # Add inference time and device info to results
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    performance_info = f"\n\nInference time: {inference_time:.3f} seconds on {device_info}"
    result_text = "\n".join(results)
    result_text += performance_info
    
    return result_text

# Define Gradio interface
with gr.Blocks(title="Object Detection Demo") as demo:
    gr.Markdown("""
    # Multi-Model Object Detection Demo
    
    This demo showcases three different object detection and image classification models:
    - **YOLOv8**: Fast and accurate object detection
    - **DETR**: DEtection TRansformer for object detection
    - **ViT**: Vision Transformer for image classification
    
    Upload an image to see how each model performs!
    """)
    
    with gr.Row():
        input_image = gr.Image(type="pil", label="Input Image")
    
    with gr.Row():
        yolo_button = gr.Button("Detect with YOLOv8")
        detr_button = gr.Button("Detect with DETR")
        vit_button = gr.Button("Classify with ViT")
    
    with gr.Row():
        with gr.Column():
            yolo_output = gr.Image(type="pil", label="YOLOv8 Detection")
            yolo_text = gr.Textbox(label="YOLOv8 Results")
        
        with gr.Column():
            detr_output = gr.Image(type="pil", label="DETR Detection")
            detr_text = gr.Textbox(label="DETR Results")
        
        with gr.Column():
            vit_text = gr.Textbox(label="ViT Classification Results")
    
    # Set up event handlers
    yolo_button.click(
        fn=process_yolo,
        inputs=input_image,
        outputs=[yolo_output, yolo_text]
    )
    
    detr_button.click(
        fn=process_detr,
        inputs=input_image,
        outputs=[detr_output, detr_text]
    )
    
    vit_button.click(
        fn=process_vit,
        inputs=input_image,
        outputs=vit_text
    )
    
   

# Launch the app
if __name__ == "__main__":
    demo.launch()
