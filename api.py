# -*- coding: utf-8 -*-
# Set matplotlib config directory to avoid permission issues
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, session, render_template_string
import torch
from PIL import Image
import numpy as np
import io
from io import BytesIO
import base64
import uuid
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from flask_cors import CORS
import json
import sys
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

# Fix for SQLite3 version compatibility with ChromaDB
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    print("Warning: pysqlite3 not found, using built-in sqlite3")

import chromadb
from chromadb.utils import embedding_functions

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# 시크릿 키 설정 (세션 암호화에 사용)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'vision_llm_agent_secret_key')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 세션 유효 시간 (초)

# Flask-Login 설정
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.session_protection = 'strong'

# 세션 설정
from flask_session import Session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['SESSION_USE_SIGNER'] = True
Session(app)

# 사용자 클래스 정의
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password
    
    def get_id(self):
        return str(self.id)  # Flask-Login은 문자열 ID를 요구함

# 테스트용 사용자 (실제 환경에서는 데이터베이스 사용 권장)
users = {
    'admin': User('1', 'admin', 'admin123'),
    'user': User('2', 'user', 'user123')
}

# 사용자 로더 함수
@login_manager.user_loader
def load_user(user_id):
    print(f"Loading user with ID: {user_id}")
    # user_id가 문자열로 전달되기 때문에 사용자 이름으로 처리
    for username, user in users.items():
        if user.id == user_id:
            print(f"User found: {username}")
            return user
    print(f"User not found with ID: {user_id}")
    return None

# Model initialization
print("Loading models... This may take a moment.")

# Image embedding model (CLIP) for vector search
clip_model = None
clip_processor = None
try:
    from transformers import CLIPProcessor, CLIPModel
    
    # 임시 디렉토리 사용
    import tempfile
    temp_dir = tempfile.gettempdir()
    os.environ["TRANSFORMERS_CACHE"] = temp_dir
    
    # CLIP 모델 로드 (이미지 임베딩용)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print("CLIP model loaded successfully")
except Exception as e:
    print("Error loading CLIP model:", e)
    clip_model = None
    clip_processor = None

# Vector DB 초기화
vector_db = None
image_collection = None
object_collection = None
try:
    # ChromaDB 클라이언트 초기화 (인메모리 DB)
    vector_db = chromadb.Client()
    
    # 임베딩 함수 설정
    ef = embedding_functions.DefaultEmbeddingFunction()
    
    # 이미지 컬렉션 생성
    image_collection = vector_db.create_collection(
        name="image_collection",
        embedding_function=ef,
        get_or_create=True
    )
    
    # 객체 인식 결과 컬렉션 생성
    object_collection = vector_db.create_collection(
        name="object_collection",
        embedding_function=ef,
        get_or_create=True
    )
    
    print("Vector DB initialized successfully")
except Exception as e:
    print("Error initializing Vector DB:", e)
    vector_db = None
    image_collection = None
    object_collection = None

# YOLOv8 model
yolo_model = None
try:
    import os
    from ultralytics import YOLO
    
    # 모델 파일 경로 - 임시 디렉토리 사용
    import tempfile
    temp_dir = tempfile.gettempdir()
    model_path = os.path.join(temp_dir, "yolov8n.pt")
    
    # 모델 파일이 없으면 직접 다운로드
    if not os.path.exists(model_path):
        print(f"Downloading YOLOv8 model to {model_path}...")
        try:
            os.system(f"wget -q https://ultralytics.com/assets/yolov8n.pt -O {model_path}")
            print("YOLOv8 model downloaded successfully")
        except Exception as e:
            print(f"Error downloading YOLOv8 model: {e}")
            # 다운로드 실패 시 대체 URL 시도
            try:
                os.system(f"wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O {model_path}")
                print("YOLOv8 model downloaded from alternative source")
            except Exception as e2:
                print(f"Error downloading from alternative source: {e2}")
                # 마지막 대안으로 직접 모델 URL 사용
                try:
                    os.system(f"curl -L https://ultralytics.com/assets/yolov8n.pt --output {model_path}")
                    print("YOLOv8 model downloaded using curl")
                except Exception as e3:
                    print(f"All download attempts failed: {e3}")
    
    # 환경 변수 설정 - 설정 파일 경로 지정
    os.environ["YOLO_CONFIG_DIR"] = temp_dir
    os.environ["MPLCONFIGDIR"] = temp_dir
    
    yolo_model = YOLO(model_path)  # Using the nano model for faster inference
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

# LLM model (using an open-access model instead of Llama 4 which requires authentication)
llm_model = None
llm_tokenizer = None
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading LLM model... This may take a moment.")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Using TinyLlama as an open-access alternative
    
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        # Removing options that require accelerate package
        # device_map="auto",
        # load_in_8bit=True
    ).to(device)
    print("LLM model loaded successfully")
except Exception as e:
    print(f"Error loading LLM model: {e}")
    llm_model = None
    llm_tokenizer = None

def process_llm_query(vision_results, user_query):
    """Process a query with the LLM model using vision results and user text"""
    if llm_model is None or llm_tokenizer is None:
        return {"error": "LLM model not available"}
    
    # 결과 데이터 요약 (토큰 길이 제한을 위해)
    summarized_results = []
    
    # 객체 탐지 결과 요약
    if isinstance(vision_results, list):
        # 최대 10개 객체만 포함
        for i, obj in enumerate(vision_results[:10]):
            if isinstance(obj, dict):
                # 필요한 정보만 추출
                summary = {
                    "label": obj.get("label", "unknown"),
                    "confidence": obj.get("confidence", 0),
                }
                summarized_results.append(summary)
    
    # Create a prompt combining vision results and user query
    prompt = f"""You are an AI assistant analyzing image detection results. 
    Here are the objects detected in the image: {json.dumps(summarized_results, indent=2)}
    
    User question: {user_query}
    
    Please provide a detailed analysis based on the detected objects and the user's question.
    """
    
    # Tokenize and generate response
    try:
        start_time = time.time()
        
        # 토큰 길이 확인 및 제한
        tokens = llm_tokenizer.encode(prompt)
        if len(tokens) > 1500:  # 안전 마진 설정
            prompt = f"""You are an AI assistant analyzing image detection results.
            The image contains {len(summarized_results)} detected objects.
            
            User question: {user_query}
            
            Please provide a general analysis based on the user's question.
            """
        
        inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = llm_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response_text = llm_tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the prompt from the response
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):].strip()
        
        inference_time = time.time() - start_time
        
        return {
            "response": response_text,
            "performance": {
                "inference_time": round(inference_time, 3),
                "device": "GPU" if torch.cuda.is_available() else "CPU"
            }
        }
    except Exception as e:
        return {"error": f"Error processing LLM query: {str(e)}"}

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
@login_required
def yolo_detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    image = Image.open(file.stream)
    
    result = process_yolo(image)
    return jsonify(result)

@app.route('/api/detect/detr', methods=['POST'])
@login_required
def detr_detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    image = Image.open(file.stream)
    
    result = process_detr(image)
    return jsonify(result)

@app.route('/api/classify/vit', methods=['POST'])
@login_required
def vit_classify():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    image = Image.open(file.stream)
    
    result = process_vit(image)
    return jsonify(result)

@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze_with_llm():
    # Check if required data is in the request
    if not request.json:
        return jsonify({"error": "No JSON data provided"}), 400
    
    # Extract vision results and user query from request
    data = request.json
    if 'visionResults' not in data or 'userQuery' not in data:
        return jsonify({"error": "Missing required fields: visionResults or userQuery"}), 400
    
    vision_results = data['visionResults']
    user_query = data['userQuery']

    # Process the query with LLM
    result = process_llm_query(vision_results, user_query)

    return jsonify(result)

def generate_image_embedding(image):
    """CLIP 모델을 사용하여 이미지 임베딩 생성"""
    if clip_model is None or clip_processor is None:
        return None

    try:
        # 이미지 전처리
        inputs = clip_processor(images=image, return_tensors="pt")

        # 이미지 임베딩 생성
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)

        # 임베딩 정규화 및 numpy 배열로 변환
        image_embedding = image_features.squeeze().cpu().numpy()
        normalized_embedding = image_embedding / np.linalg.norm(image_embedding)

        return normalized_embedding.tolist()
    except Exception as e:
        print(f"Error generating image embedding: {e}")
        return None

@app.route('/api/similar-images', methods=['POST'])
@login_required
def find_similar_images():
    """유사 이미지 검색 API"""
    if clip_model is None or clip_processor is None or image_collection is None:
        return jsonify({"error": "Image embedding model or vector DB not available"})

    try:
        # 요청에서 이미지 데이터 추출
        if 'image' not in request.files and 'image' not in request.form:
            return jsonify({"error": "No image provided"})

        if 'image' in request.files:
            # 파일로 업로드된 경우
            image_file = request.files['image']
            image = Image.open(image_file).convert('RGB')
        else:
            # base64로 인코딩된 경우
            image_data = request.form['image']
            if image_data.startswith('data:image'):
                # Remove the data URL prefix if present
                image_data = image_data.split(',')[1]
            image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')

        # 이미지 ID 생성 (임시)
        image_id = str(uuid.uuid4())

        # 이미지 임베딩 생성
        embedding = generate_image_embedding(image)
        if embedding is None:
            return jsonify({"error": "Failed to generate image embedding"})

        # 현재 이미지를 DB에 추가 (선택적)
        # image_collection.add(
        #    ids=[image_id],
        #    embeddings=[embedding]
        # )

        # 유사 이미지 검색
        results = image_collection.query(
            query_embeddings=[embedding],
            n_results=5  # 상위 5개 결과 반환
        )

        # 결과 포맷팅
        similar_images = []
        if len(results['ids'][0]) > 0:
            for i, img_id in enumerate(results['ids'][0]):
                similar_images.append({
                    "id": img_id,
                    "distance": float(results['distances'][0][i]) if 'distances' in results else 0.0,
                    "metadata": results['metadatas'][0][i] if 'metadatas' in results else {}
                })

        return jsonify({
            "query_image_id": image_id,
            "similar_images": similar_images
        })

    except Exception as e:
        print(f"Error in similar-images API: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/add-to-collection', methods=['POST'])
@login_required
def add_to_collection():
    """이미지를 벡터 DB에 추가하는 API"""
    if clip_model is None or clip_processor is None or image_collection is None:
        return jsonify({"error": "Image embedding model or vector DB not available"})

    try:
        # 요청에서 이미지 데이터 추출
        if 'image' not in request.files and 'image' not in request.form:
            return jsonify({"error": "No image provided"})

        # 메타데이터 추출
        metadata = {}
        if 'metadata' in request.form:
            metadata = json.loads(request.form['metadata'])

        # 이미지 ID (제공되지 않은 경우 자동 생성)
        image_id = request.form.get('id', str(uuid.uuid4()))

        if 'image' in request.files:
            # 파일로 업로드된 경우
            image_file = request.files['image']
            image = Image.open(image_file).convert('RGB')
        else:
            # base64로 인코딩된 경우
            image_data = request.form['image']
            if image_data.startswith('data:image'):
                # Remove the data URL prefix if present
                image_data = image_data.split(',')[1]
            image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')

        # 이미지 임베딩 생성
        embedding = generate_image_embedding(image)
        if embedding is None:
            return jsonify({"error": "Failed to generate image embedding"})
            
        # 이미지 데이터를 base64로 인코딩하여 메타데이터에 추가
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        metadata['image_data'] = img_str

        # 이미지를 DB에 추가
        image_collection.add(
            ids=[image_id],
            embeddings=[embedding],
            metadatas=[metadata]
        )

        return jsonify({
            "success": True,
            "image_id": image_id,
            "message": "Image added to collection"
        })

    except Exception as e:
        print(f"Error in add-to-collection API: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/add-detected-objects', methods=['POST'])
@login_required
def add_detected_objects():
    """객체 인식 결과를 벡터 DB에 추가하는 API"""
    if clip_model is None or object_collection is None:
        return jsonify({"error": "Image embedding model or vector DB not available"})

    try:
        # 디버깅: 요청 데이터 로깅
        print("[DEBUG] Received request in add-detected-objects")
        
        # 요청에서 이미지와 객체 검출 결과 데이터 추출
        data = request.json
        print(f"[DEBUG] Request data keys: {list(data.keys()) if data else 'None'}")
        
        if not data:
            print("[DEBUG] Error: No data received in request")
            return jsonify({"error": "No data received"})
            
        if 'image' not in data:
            print("[DEBUG] Error: 'image' key not found in request data")
            return jsonify({"error": "Missing image data"})
            
        if 'objects' not in data:
            print("[DEBUG] Error: 'objects' key not found in request data")
            return jsonify({"error": "Missing objects data"})
        
        # 이미지 데이터 디버깅
        print(f"[DEBUG] Image data type: {type(data['image'])}")
        print(f"[DEBUG] Image data starts with: {data['image'][:50]}...") # 처음 50자만 출력
        
        # 객체 데이터 디버깅
        print(f"[DEBUG] Objects data type: {type(data['objects'])}")
        print(f"[DEBUG] Objects count: {len(data['objects']) if isinstance(data['objects'], list) else 'Not a list'}")
        if isinstance(data['objects'], list) and len(data['objects']) > 0:
            print(f"[DEBUG] First object keys: {list(data['objects'][0].keys()) if isinstance(data['objects'][0], dict) else 'Not a dict'}")
        
        # 이미지 데이터 처리
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
        image_width, image_height = image.size
        
        # 이미지 ID
        image_id = data.get('imageId', str(uuid.uuid4()))
        
        # 객체 데이터 처리
        objects = data['objects']
        object_ids = []
        object_embeddings = []
        object_metadatas = []
        
        for obj in objects:
            # 객체 ID 생성
            object_id = f"{image_id}_{str(uuid.uuid4())[:8]}"
            
            # 바운딩 박스 정보 추출
            bbox = obj.get('bbox', [])
            
            # 리스트 형태의 bbox [x1, y1, x2, y2] 처리
            if isinstance(bbox, list) and len(bbox) >= 4:
                x1 = bbox[0] / image_width  # 정규화된 좌표로 변환
                y1 = bbox[1] / image_height
                x2 = bbox[2] / image_width
                y2 = bbox[3] / image_height
                width = x2 - x1
                height = y2 - y1
            # 딕셔너리 형태의 bbox {'x': x, 'y': y, 'width': width, 'height': height} 처리
            elif isinstance(bbox, dict):
                x1 = bbox.get('x', 0)
                y1 = bbox.get('y', 0)
                width = bbox.get('width', 0)
                height = bbox.get('height', 0)
            else:
                # 기본값 설정
                x1, y1, width, height = 0, 0, 0, 0
            
            # 바운딩 박스를 이미지 좌표로 변환
            x1_px = int(x1 * image_width)
            y1_px = int(y1 * image_height)
            width_px = int(width * image_width)
            height_px = int(height * image_height)
            
            # 객체 이미지 자르기
            try:
                object_image = image.crop((x1_px, y1_px, x1_px + width_px, y1_px + height_px))
                
                # 임베딩 생성
                embedding = generate_image_embedding(object_image)
                if embedding is None:
                    continue
                
                # 메타데이터 구성
                # bbox를 JSON 문자열로 직렬화하여 ChromaDB 메타데이터 제한 우회
                bbox_json = json.dumps({
                    "x": x1,
                    "y": y1,
                    "width": width,
                    "height": height
                })
                
                # 객체 이미지를 base64로 인코딩
                buffered = BytesIO()
                object_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                metadata = {
                    "image_id": image_id,
                    "class": obj.get('class', ''),
                    "confidence": obj.get('confidence', 0),
                    "bbox": bbox_json,  # JSON 문자열로 저장
                    "image_data": img_str  # 이미지 데이터 추가
                }
                
                object_ids.append(object_id)
                object_embeddings.append(embedding)
                object_metadatas.append(metadata)
            except Exception as e:
                print(f"Error processing object: {e}")
                continue
        
        # 객체가 없는 경우
        if not object_ids:
            return jsonify({"error": "No valid objects to add"})
        
        # 디버깅: 메타데이터 출력
        print(f"[DEBUG] Adding {len(object_ids)} objects to vector DB")
        print(f"[DEBUG] First metadata sample: {object_metadatas[0] if object_metadatas else 'None'}")
        
        try:
            # 객체들을 DB에 추가
            object_collection.add(
                ids=object_ids,
                embeddings=object_embeddings,
                metadatas=object_metadatas
            )
            print("[DEBUG] Successfully added objects to vector DB")
        except Exception as e:
            print(f"[DEBUG] Error adding to vector DB: {e}")
            raise e
        
        return jsonify({
            "success": True,
            "image_id": image_id,
            "object_count": len(object_ids),
            "object_ids": object_ids
        })
    
    except Exception as e:
        print(f"Error in add-detected-objects API: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/search-similar-objects', methods=['POST'])
@login_required
def search_similar_objects():
    """유사한 객체 검색 API"""
    print("[DEBUG] Received request in search-similar-objects")
    
    if clip_model is None or object_collection is None:
        print("[DEBUG] Error: Image embedding model or vector DB not available")
        return jsonify({"error": "Image embedding model or vector DB not available"})

    try:
        # 요청 데이터 추출
        data = request.json
        print(f"[DEBUG] Request data keys: {list(data.keys()) if data else 'None'}")
        
        if not data:
            print("[DEBUG] Error: Missing request data")
            return jsonify({"error": "Missing request data"})
        
        # 검색 유형 결정
        search_type = data.get('searchType', 'image')
        n_results = int(data.get('n_results', 5))  # 결과 개수
        print(f"[DEBUG] Search type: {search_type}, n_results: {n_results}")
        
        query_embedding = None
        
        if search_type == 'image' and 'image' in data:
            # 이미지로 검색하는 경우
            print("[DEBUG] Searching by image")
            image_data = data['image']
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            try:
                image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
                query_embedding = generate_image_embedding(image)
                print(f"[DEBUG] Generated image embedding: {type(query_embedding)}, shape: {len(query_embedding) if query_embedding is not None else 'None'}")
            except Exception as e:
                print(f"[DEBUG] Error generating image embedding: {e}")
                return jsonify({"error": f"Error processing image: {str(e)}"}), 500
        
        elif search_type == 'object' and 'objectId' in data:
            # 객체 ID로 검색하는 경우
            object_id = data['objectId']
            result = object_collection.get(ids=[object_id], include=["embeddings"])
            
            if result and "embeddings" in result and len(result["embeddings"]) > 0:
                query_embedding = result["embeddings"][0]
        
        elif search_type == 'class' and 'class_name' in data:
            # 클래스 이름으로 검색하는 경우
            print("[DEBUG] Searching by class name")
            class_name = data['class_name']
            print(f"[DEBUG] Class name: {class_name}")
            filter_query = {"class": {"$eq": class_name}}
            
            try:
                # 클래스로 필터링하여 검색
                print(f"[DEBUG] Querying with filter: {filter_query}")
                # Use get method instead of query for class-based filtering
                results = object_collection.get(
                    where=filter_query,
                    limit=n_results,
                    include=["metadatas", "embeddings", "documents"]
                )
                
                print(f"[DEBUG] Query results: {results['ids'][0] if 'ids' in results and len(results['ids']) > 0 else 'No results'}")
                formatted_results = format_object_results(results)
                print(f"[DEBUG] Formatted results count: {len(formatted_results)}")
                
                return jsonify({
                    "success": True,
                    "searchType": "class",
                    "results": formatted_results
                })
            except Exception as e:
                print(f"[DEBUG] Error in class search: {e}")
                return jsonify({"error": f"Error in class search: {str(e)}"}), 500
        
        else:
            print(f"[DEBUG] Invalid search parameters: {data}")
            return jsonify({"error": "Invalid search parameters"})
        
        if query_embedding is None:
            print("[DEBUG] Error: Failed to generate query embedding")
            return jsonify({"error": "Failed to generate query embedding"})
        
        try:
            # 유사도 검색 실행
            print(f"[DEBUG] Running similarity search with embedding of length {len(query_embedding)}")
            results = object_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "distances"]
            )
            
            print(f"[DEBUG] Query results: {results['ids'][0] if 'ids' in results and len(results['ids']) > 0 else 'No results'}")
            formatted_results = format_object_results(results)
            print(f"[DEBUG] Formatted results count: {len(formatted_results)}")
            
            return jsonify({
                "success": True,
                "searchType": search_type,
                "results": formatted_results
            })
        except Exception as e:
            print(f"[DEBUG] Error in similarity search: {e}")
            return jsonify({"error": f"Error in similarity search: {str(e)}"}), 500
    
    except Exception as e:
        print(f"Error in search-similar-objects API: {e}")
        return jsonify({"error": str(e)}), 500

def format_object_results(results):
    """검색 결과 포맷팅 - ChromaDB query 및 get 메서드 결과 모두 처리"""
    formatted_results = []
    
    print(f"[DEBUG] Formatting results: {results.keys() if results else 'None'}")
    
    if not results:
        print("[DEBUG] No results to format")
        return formatted_results
    
    try:
        # Check if this is a result from 'get' method (class search) or 'query' method (similarity search)
        is_get_result = 'ids' in results and isinstance(results['ids'], list) and not isinstance(results['ids'][0], list) if 'ids' in results else False
        
        if is_get_result:
            # Handle results from 'get' method (flat structure)
            print("[DEBUG] Processing results from get method (class search)")
            if len(results['ids']) == 0:
                return formatted_results
                
            for i, obj_id in enumerate(results['ids']):
                try:
                    # Extract object info
                    metadata = results['metadatas'][i] if 'metadatas' in results else {}
                    
                    # Deserialize bbox if stored as JSON string
                    if 'bbox' in metadata and isinstance(metadata['bbox'], str):
                        try:
                            metadata['bbox'] = json.loads(metadata['bbox'])
                        except:
                            pass  # Keep as is if deserialization fails
                    
                    result_item = {
                        "id": obj_id,
                        "metadata": metadata
                    }
                    
                    # No distance in get results
                    
                    # Check if image data is already in metadata
                    if 'image_data' not in metadata:
                        print(f"[DEBUG] Image data not found in metadata for object {obj_id}")
                    else:
                        print(f"[DEBUG] Image data found in metadata for object {obj_id}")
                    
                    formatted_results.append(result_item)
                except Exception as e:
                    print(f"[DEBUG] Error formatting get result {i}: {e}")
        else:
            # Handle results from 'query' method (nested structure)
            print("[DEBUG] Processing results from query method (similarity search)")
            if 'ids' not in results or len(results['ids']) == 0 or len(results['ids'][0]) == 0:
                return formatted_results
                
            for i, obj_id in enumerate(results['ids'][0]):
                try:
                    # Extract object info
                    metadata = results['metadatas'][0][i] if 'metadatas' in results and len(results['metadatas']) > 0 else {}
                    
                    # Deserialize bbox if stored as JSON string
                    if 'bbox' in metadata and isinstance(metadata['bbox'], str):
                        try:
                            metadata['bbox'] = json.loads(metadata['bbox'])
                        except:
                            pass  # Keep as is if deserialization fails
                    
                    result_item = {
                        "id": obj_id,
                        "metadata": metadata
                    }
                    
                    if 'distances' in results and len(results['distances']) > 0:
                        result_item["distance"] = float(results['distances'][0][i])
                    
                    # Check if image data is already in metadata
                    if 'image_data' not in metadata:
                        try:
                            # Try to get original image via image ID
                            image_id = metadata.get('image_id')
                            if image_id:
                                print(f"[DEBUG] Image data not found in metadata for object {obj_id} with image_id {image_id}")
                        except Exception as e:
                            print(f"[DEBUG] Error checking image data for result {i}: {e}")
                    else:
                        print(f"[DEBUG] Image data found in metadata for object {obj_id}")
                    
                    formatted_results.append(result_item)
                except Exception as e:
                    print(f"[DEBUG] Error formatting query result {i}: {e}")
    except Exception as e:
        print(f"[DEBUG] Error in format_object_results: {e}")
    
    return formatted_results

# 로그인 페이지 HTML 템플릿
LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vision LLM Agent - 로그인</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .login-container {
            background-color: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 400px;
        }
        h1 {
            text-align: center;
            color: #4a6cf7;
            margin-bottom: 1.5rem;
        }
        .form-group {
            margin-bottom: 1rem;
        }
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: bold;
        }
        input {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 1rem;
        }
        button {
            width: 100%;
            padding: 0.75rem;
            background-color: #4a6cf7;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 1rem;
            cursor: pointer;
            margin-top: 1rem;
        }
        button:hover {
            background-color: #3a5cd8;
        }
        .error-message {
            color: #e74c3c;
            margin-top: 1rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h1>Vision LLM Agent</h1>
        <form action="/login" method="post">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit">Login</button>
            {% if error %}
            <p class="error-message">{{ error }}</p>
            {% endif %}
        </form>
    </div>
</body>
</html>
'''

@app.route('/login', methods=['GET', 'POST'])
def login():
    # 이미 로그인된 사용자는 메인 페이지로 리디렉션
    if current_user.is_authenticated:
        print(f"User already authenticated as: {current_user.username}, redirecting to index")
        return redirect('/index.html')
    
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        print(f"Login attempt: username={username}")
        
        if username in users and users[username].password == password:
            # 로그인 성공 시 세션에 사용자 정보 저장
            user = users[username]
            login_user(user, remember=True)
            session['user_id'] = user.id
            session['username'] = username
            session.permanent = True
            
            print(f"Login successful for user: {username}, ID: {user.id}")
            
            # 리디렉션 처리
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/') and next_page != '/login':
                print(f"Redirecting to: {next_page}")
                return redirect(next_page)
            print("Redirecting to index.html")
            return redirect('/index.html')
        else:
            error = 'Invalid username or password'
            print(f"Login failed: {error}")
    
    return render_template_string(LOGIN_TEMPLATE, error=error)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

# 정적 파일 서빙을 위한 라우트 (로그인 불필요)
@app.route('/static/<path:filename>')
def serve_static(filename):
    print(f"Serving static file: {filename}")
    return send_from_directory(app.static_folder, filename)

# 인덱스 HTML 직접 서빙 (로그인 필요)
@app.route('/index.html')
@login_required
def serve_index_html():
    print(f"Serving index.html for user: {current_user.username if current_user.is_authenticated else 'not authenticated'}")
    return send_from_directory(app.static_folder, 'index.html')

# 기본 경로 및 기타 경로 처리 (로그인 필요)
@app.route('/', defaults={'path': ''}, methods=['GET'])
@app.route('/<path:path>', methods=['GET'])
@login_required
def serve_react(path):
    """Serve React frontend"""
    print(f"Serving React frontend for path: {path}, user: {current_user.username if current_user.is_authenticated else 'not authenticated'}")
    # 정적 파일 처리는 이제 별도 라우트에서 처리
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        # React 앱의 index.html 서빙
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/similar-images', methods=['GET'])
@login_required
def similar_images_page():
    """Serve similar images search page"""
    return send_from_directory(app.static_folder, 'similar-images.html')

@app.route('/object-detection-search', methods=['GET'])
@login_required
def object_detection_search_page():
    """Serve object detection search page"""
    return send_from_directory(app.static_folder, 'object-detection-search.html')

@app.route('/model-vector-db', methods=['GET'])
@login_required
def model_vector_db_page():
    """Serve model vector DB UI page"""
    return send_from_directory(app.static_folder, 'model-vector-db.html')

@app.route('/api/status', methods=['GET'])
@login_required
def status():
    return jsonify({
        "status": "online",
        "models": {
            "yolo": yolo_model is not None,
            "detr": detr_model is not None and detr_processor is not None,
            "vit": vit_model is not None and vit_processor is not None
        },
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "user": current_user.username
    })

# Root route is now handled by serve_react function
# This route is removed to prevent conflicts

@app.route('/index')
@login_required
def index_page():
    # /index 경로는 index.html로 리디렉션
    print("Index route redirecting to index.html")
    return redirect('/index.html')

if __name__ == "__main__":
    # 허깅페이스 Space에서는 PORT 환경 변수를 사용합니다
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host='0.0.0.0', port=port)
