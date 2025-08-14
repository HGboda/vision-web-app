import gradio as gr
import torch
from PIL import Image
import numpy as np
import os
import requests
import json
import base64
from io import BytesIO
import uuid

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

# 벡터 DB에 객체 저장 함수
def save_objects_to_vector_db(image, detection_results, model_type='yolo'):
    if image is None or detection_results is None:
        return "이미지나 객체 인식 결과가 없습니다."
    
    try:
        # 이미지를 base64로 인코딩
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # 모델 타입에 따라 다른 API 엔드포인트 호출
        if model_type in ['yolo', 'detr']:
            # 객체 정보 추출
            objects = []
            for obj in detection_results['objects']:
                objects.append({
                    "class": obj['class'],
                    "confidence": obj['confidence'],
                    "bbox": obj['bbox']
                })
            
            # API 요청 데이터 구성
            data = {
                "image": img_str,
                "objects": objects,
                "image_id": str(uuid.uuid4())
            }
            
            # API 호출
            response = requests.post(
                "http://localhost:7860/api/add-detected-objects", 
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'error' in result:
                    return f"오류 발생: {result['error']}"
                return f"벡터 DB에 {len(objects)}개 객체 저장 완료! ID: {result.get('ids', '알 수 없음')}"
        
        elif model_type == 'vit':
            # ViT 분류 결과 저장
            data = {
                "image": img_str,
                "metadata": {
                    "model": "vit",
                    "classifications": detection_results.get('classifications', [])
                }
            }
            
            # API 호출
            response = requests.post(
                "http://localhost:7860/api/add-image", 
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'error' in result:
                    return f"오류 발생: {result['error']}"
                return f"벡터 DB에 이미지 및 분류 결과 저장 완료! ID: {result.get('id', '알 수 없음')}"
        
        else:
            return "지원하지 않는 모델 타입입니다."
            
        if response.status_code != 200:
            return f"API 오류: {response.status_code}"
    except Exception as e:
        return f"오류 발생: {str(e)}"

# 벡터 DB에서 유사 객체 검색 함수
def search_similar_objects(image=None, class_name=None):
    try:
        data = {}
        
        if image is not None:
            # 이미지를 base64로 인코딩
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            data["image"] = img_str
            data["n_results"] = 5
        elif class_name is not None and class_name.strip():
            data["class_name"] = class_name.strip()
            data["n_results"] = 5
        else:
            return "이미지나 클래스 이름 중 하나는 제공해야 합니다.", []
        
        # API 호출
        response = requests.post(
            "http://localhost:7860/api/search-similar-objects", 
            json=data
        )
        
        if response.status_code == 200:
            results = response.json()
            if isinstance(results, dict) and 'error' in results:
                return f"오류 발생: {results['error']}", []
                
            # 결과 포맷팅
            formatted_results = []
            for i, result in enumerate(results):
                similarity = (1 - result.get('distance', 0)) * 100
                img_data = result.get('image', '')
                
                # 이미지 데이터를 PIL 이미지로 변환
                if img_data:
                    try:
                        img_bytes = base64.b64decode(img_data)
                        img = Image.open(BytesIO(img_bytes))
                    except Exception:
                        img = None
                else:
                    img = None
                
                # 메타데이터 추출
                metadata = result.get('metadata', {})
                class_name = metadata.get('class', 'N/A')
                confidence = metadata.get('confidence', 0) * 100 if metadata.get('confidence') else 'N/A'
                
                formatted_results.append({
                    'image': img,
                    'info': f"결과 #{i+1} | 유사도: {similarity:.2f}% | 클래스: {class_name} | 신뢰도: {confidence if isinstance(confidence, str) else f'{confidence:.2f}%'} | ID: {result.get('id', 'N/A')}"
                })
                
            return f"{len(formatted_results)}개의 유사 객체를 찾았습니다.", formatted_results
        else:
            return f"API 오류: {response.status_code}", []
    except Exception as e:
        return f"오류 발생: {str(e)}", []

# Define model inference functions
def process_yolo(image):
    if yolo_model is None:
        return None, "YOLOv8 model not loaded", None
    
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
    detection_objects = {'objects': []}
    
    for box in boxes:
        class_id = int(box.cls[0].item())
        class_name = class_names[class_id]
        confidence = round(box.conf[0].item(), 2)
        bbox = box.xyxy[0].tolist()
        bbox = [round(x) for x in bbox]
        
        detections.append("{}: {} at {}".format(class_name, confidence, bbox))
        
        # 벡터 DB 저장용 객체 정보 추가
        detection_objects['objects'].append({
            'class': class_name,
            'confidence': confidence,
            'bbox': bbox
        })
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    # Add inference time and device info to detection text
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    performance_info = f"\n\nInference time: {inference_time:.3f} seconds on {device_info}"
    detection_text = "\n".join(detections) if detections else "No objects detected"
    detection_text += performance_info
    
    return result_image, detection_text, detection_objects
    
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
    
    # 벡터 DB 저장 버튼 및 결과 표시
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 벡터 DB 저장")
            save_yolo_button = gr.Button("YOLOv8 인식 결과 저장", variant="primary")
            save_detr_button = gr.Button("DETR 인식 결과 저장", variant="primary")
            save_vit_button = gr.Button("ViT 분류 결과 저장", variant="primary")
            save_result = gr.Textbox(label="벡터 DB 저장 결과")
        
        with gr.Column():
            gr.Markdown("### 벡터 DB 검색")
            search_class = gr.Textbox(label="클래스 이름으로 검색")
            search_button = gr.Button("검색", variant="secondary")
            search_result_text = gr.Textbox(label="검색 결과 정보")
            search_result_gallery = gr.Gallery(label="검색 결과", columns=5, height=300)
    
    # 객체 인식 결과 저장용 상태 변수
    yolo_detection_state = gr.State(None)
    detr_detection_state = gr.State(None)
    vit_classification_state = gr.State(None)
    
    # Set up event handlers
    yolo_button.click(
        fn=process_yolo,
        inputs=input_image,
        outputs=[yolo_output, yolo_text, yolo_detection_state]
    )
    
    # DETR 결과 처리 함수 수정 - 상태 저장 추가
    def process_detr_with_state(image):
        result_image, result_text = process_detr(image)
        
        # 객체 인식 결과 추출
        detection_results = {"objects": []}
        
        # 결과 텍스트에서 객체 정보 추출
        lines = result_text.split('\n')
        for line in lines:
            if ': ' in line and ' at ' in line:
                try:
                    class_conf, location = line.split(' at ')
                    class_name, confidence = class_conf.split(': ')
                    confidence = float(confidence)
                    
                    # 바운딩 박스 정보 추출
                    bbox_str = location.strip('[]').split(', ')
                    bbox = [int(coord) for coord in bbox_str]
                    
                    detection_results["objects"].append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": bbox
                    })
                except Exception:
                    pass
        
        return result_image, result_text, detection_results
    
    # ViT 결과 처리 함수 수정 - 상태 저장 추가
    def process_vit_with_state(image):
        result_text = process_vit(image)
        
        # 분류 결과 추출
        classifications = []
        
        # 결과 텍스트에서 분류 정보 추출
        lines = result_text.split('\n')
        for line in lines:
            if '. ' in line and ': ' in line:
                try:
                    rank_class, confidence = line.split(': ')
                    _, class_name = rank_class.split('. ')
                    confidence = float(confidence)
                    
                    classifications.append({
                        "class": class_name,
                        "confidence": confidence
                    })
                except Exception:
                    pass
        
        return result_text, {"classifications": classifications}
    
    detr_button.click(
        fn=process_detr_with_state,
        inputs=input_image,
        outputs=[detr_output, detr_text, detr_detection_state]
    )
    
    vit_button.click(
        fn=process_vit_with_state,
        inputs=input_image,
        outputs=[vit_text, vit_classification_state]
    )
    
    # 벡터 DB 저장 버튼 이벤트 핸들러
    save_yolo_button.click(
        fn=lambda img, det: save_objects_to_vector_db(img, det, 'yolo'),
        inputs=[input_image, yolo_detection_state],
        outputs=save_result
    )
    
    save_detr_button.click(
        fn=lambda img, det: save_objects_to_vector_db(img, det, 'detr'),
        inputs=[input_image, detr_detection_state],
        outputs=save_result
    )
    
    save_vit_button.click(
        fn=lambda img, det: save_objects_to_vector_db(img, det, 'vit'),
        inputs=[input_image, vit_classification_state],
        outputs=save_result
    )
    
    # 검색 버튼 이벤트 핸들러
    def format_search_results(result_text, results):
        images = []
        captions = []
        
        for result in results:
            if result.get('image'):
                images.append(result['image'])
                captions.append(result['info'])
        
        return result_text, [(img, cap) for img, cap in zip(images, captions)]
    
    search_button.click(
        fn=lambda class_name: search_similar_objects(class_name=class_name),
        inputs=search_class,
        outputs=[search_result_text, search_result_gallery]
    )
    
   

# Launch the app
if __name__ == "__main__":
    demo.launch()
