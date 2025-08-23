# -*- coding: utf-8 -*-
# Set matplotlib config directory to avoid permission issues
import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, session, render_template_string, make_response, Response, stream_with_context
from datetime import timedelta
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
import requests
import asyncio
from threading import Thread
try:
    from openai import OpenAI
except Exception as _e:
    OpenAI = None
try:
    # LangChain for RAG answering
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except Exception as _e:
    ChatOpenAI = None
    ChatPromptTemplate = None
    StrOutputParser = None
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
    fresh_login_required,
    login_fresh,
)

# Fix for SQLite3 version compatibility with ChromaDB
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    print("Warning: pysqlite3 not found, using built-in sqlite3")

import chromadb
from chromadb.utils import embedding_functions

app = Flask(__name__, static_folder='static')

# Import product comparison coordinator
try:
    from product_comparison import get_product_comparison_coordinator, decode_base64_image
except ImportError:
    print("Warning: Product comparison module not available")
    get_product_comparison_coordinator = None
    decode_base64_image = None
# 환경 변수에서 비밀 키를 가져오거나, 없으면 안전한 랜덤 키 생성
secret_key = os.environ.get('FLASK_SECRET_KEY')
if not secret_key:
    import secrets
    secret_key = secrets.token_hex(16)  # 32자 길이의 랜덤 16진수 문자열 생성
    print("WARNING: FLASK_SECRET_KEY 환경 변수가 설정되지 않았습니다. 랜덤 키를 생성했습니다.")
    print("서버 재시작 시 세션이 모두 만료됩니다. 프로덕션 환경에서는 환경 변수를 설정하세요.")
app.secret_key = secret_key  # 세션 암호화를 위한 비밀 키
app.config['CORS_HEADERS'] = 'Content-Type'
# Remember cookie (Flask-Login) — minimize duration to prevent auto re-login
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(seconds=1)
app.config['REMEMBER_COOKIE_SECURE'] = True  # Spaces uses HTTPS
app.config['REMEMBER_COOKIE_HTTPONLY'] = True
app.config['REMEMBER_COOKIE_SAMESITE'] = 'None'
# Session cookie (Flask-Session)
app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_PATH'] = '/'
CORS(app)  # Enable CORS for all routes

# 시크릿 키 설정 (세션 암호화에 사용)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'vision_llm_agent_secret_key')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=120)  # 세션 유효 시간 (2분)
app.config['SESSION_REFRESH_EACH_REQUEST'] = False  # 절대 만료(로그인 기준 2분 후 만료)

# Flask-Login 설정
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.session_protection = 'strong'

# When authentication is required or session is not fresh, redirect to login instead of 401
login_manager.refresh_view = 'login'

@login_manager.unauthorized_handler
def handle_unauthorized():
    # For non-authenticated access, send user to login
    return redirect(url_for('login'))

@login_manager.needs_refresh_handler
def handle_needs_refresh():
    # For non-fresh sessions (e.g., after expiry or only remember-cookie), send to login
    return redirect(url_for('login'))

# 세션 설정
import tempfile
from flask_session import Session

# 임시 디렉토리를 사용하여 권한 문제 해결
session_dir = tempfile.gettempdir()
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_FILE_DIR'] = session_dir
print(f"Using session directory: {session_dir}")
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
# 환경 변수에서 사용자 계정 정보를 가져오기 (기본값 없음)
admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
admin_password = os.environ.get('ADMIN_PASSWORD')
user_username = os.environ.get('USER_USERNAME', 'user')
user_password = os.environ.get('USER_PASSWORD')

# 환경 변수가 설정되지 않았을 경우 경고 메시지 출력
if not admin_password or not user_password:
    print("ERROR: 환경 변수 ADMIN_PASSWORD 또는 USER_PASSWORD가 설정되지 않았습니다.")
    print("Hugging Face Spaces에서 반드시 환경 변수를 설정해야 합니다.")
    print("Settings > Repository secrets에서 환경 변수를 추가하세요.")
    # 환경 변수가 없을 경우 임시 비밀번호 생성 (개발용)
    import secrets
    if not admin_password:
        admin_password = secrets.token_hex(8)  # 임시 비밀번호 생성
        print(f"WARNING: 임시 admin 비밀번호가 생성되었습니다: {admin_password}")
    if not user_password:
        user_password = secrets.token_hex(8)  # 임시 비밀번호 생성
        print(f"WARNING: 임시 user 비밀번호가 생성되었습니다: {user_password}")

users = {
    admin_username: User('1', admin_username, admin_password),
    user_username: User('2', user_username, user_password)
}

# 사용자 로더 함수
@login_manager.user_loader
def load_user(user_id):
    print(f"Loading user with ID: {user_id}")
    # 세션 디버그 정보 출력
    print(f"Session data in user_loader: {dict(session)}")
    print(f"Current request cookies: {request.cookies}")
    
    # user_id가 문자열로 전달되기 때문에 사용자 ID로 처리
    for username, user in users.items():
        if str(user.id) == str(user_id):  # 확실한 문자열 비교
            print(f"User found: {username}, ID: {user.id}")
            # 세션 정보 업데이트
            session['user_id'] = user.id
            session['username'] = username
            session.modified = True
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
        
        # Add to vector DB
        if object_ids and object_embeddings and object_metadatas:
            object_collection.add(
                ids=object_ids,
                embeddings=object_embeddings,
                metadatas=object_metadatas
            )
            
            return jsonify({
                "success": True,
                "message": f"Added {len(object_ids)} objects to collection",
                "object_ids": object_ids
            })
        else:
            return jsonify({
                "warning": "No valid objects to add"
            })
            
    except Exception as e:
        print(f"Error in add-detected-objects API: {e}")
        return jsonify({"error": str(e)}), 500


# Product Comparison API Endpoints
@app.route('/api/product/compare/start', methods=['POST'])
@login_required
def start_product_comparison():
    """Start a new product comparison session"""
    if get_product_comparison_coordinator is None:
        return jsonify({"error": "Product comparison module not available"}), 500
    
    try:
        # Generate session ID if provided in form or query params, otherwise create new one
        session_id = request.form.get('session_id') or request.args.get('session_id') or str(uuid.uuid4())
        
        # Get analysis type if provided (info, compare, value, recommend)
        analysis_type = request.form.get('analysisType') or request.args.get('analysisType', 'info')
        
        # Process images from FormData or JSON
        images = []
        
        # Check if request is multipart form data
        if request.files:
            # Handle FormData with file uploads (from frontend)
            if 'image1' in request.files and request.files['image1']:
                img1 = request.files['image1']
                try:
                    images.append(Image.open(img1.stream))
                except Exception as e:
                    print(f"Error processing image1: {e}")
                    
            if 'image2' in request.files and request.files['image2']:
                img2 = request.files['image2']
                try:
                    images.append(Image.open(img2.stream))
                except Exception as e:
                    print(f"Error processing image2: {e}")
                    
        # Fallback to JSON with base64 images (for API testing)
        elif request.json and 'images' in request.json:
            image_data_list = request.json.get('images', [])
            for image_data in image_data_list:
                img = decode_base64_image(image_data)
                if img is not None:
                    images.append(img)
                    
        if not images:
            return jsonify({"error": "No valid images provided"}), 400
        
        # Get coordinator instance
        coordinator = get_product_comparison_coordinator()
        
        # Pass the analysis type and session metadata to the coordinator
        session_metadata = {
            'analysis_type': analysis_type,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Start processing in a background thread
        def run_async_task(loop):
            asyncio.set_event_loop(loop)
            loop.run_until_complete(coordinator.process_images(session_id, images, session_metadata))
        
        loop = asyncio.new_event_loop()
        thread = Thread(target=run_async_task, args=(loop,))
        thread.daemon = True
        thread.start()
        
        # Return session ID for client to use with streaming endpoint
        return jsonify({
            "session_id": session_id,
            "message": "Product comparison started",
            "status": "processing"
        })
        
    except Exception as e:
        print(f"Error starting product comparison: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/product/compare/stream/<session_id>', methods=['GET'])
@login_required
def stream_product_comparison(session_id):
    """Stream updates from a product comparison session"""
    if get_product_comparison_coordinator is None:
        return jsonify({"error": "Product comparison module not available"}), 500
    
    def generate():
        """Generate SSE events for streaming"""
        coordinator = get_product_comparison_coordinator()
        last_message_index = 0
        retry_count = 0
        max_retries = 300  # 5 minutes at 1 second intervals
        
        while retry_count < max_retries:
            # Get current status
            status = coordinator.get_session_status(session_id)
            if status is None:
                # Session not found
                yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
                break
            
            # Get all messages
            messages = coordinator.get_session_messages(session_id)
            
            # Send any new messages
            if messages and len(messages) > last_message_index:
                new_messages = messages[last_message_index:]
                for msg in new_messages:
                    yield f"data: {json.dumps({'message': msg})}\n\n"
                last_message_index = len(messages)
            
            # Send current status
            yield f"data: {json.dumps({'status': status})}\n\n"
            
            # If completed or error, send final result and end stream
            if status in ['completed', 'error']:
                result = coordinator.get_session_result(session_id)
                yield f"data: {json.dumps({'final_result': result})}\n\n"
                break
            
            # Wait before next update
            time.sleep(1)
            retry_count += 1
        
        # End the stream if we've reached max retries
        if retry_count >= max_retries:
            yield f"data: {json.dumps({'error': 'Timeout waiting for results'})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Content-Type': 'text/event-stream',
        }
    )
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
        <form action="/login" method="post" autocomplete="off">
            <!-- hidden dummy fields to discourage Chrome autofill -->
            <input type="text" name="fakeusernameremembered" style="display:none" tabindex="-1" autocomplete="off">
            <input type="password" name="fakepasswordremembered" style="display:none" tabindex="-1" autocomplete="off">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" value="user" required autocomplete="username" autocapitalize="none" autocorrect="off" spellcheck="false">
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" value="user123" placeholder="******" required autocomplete="current-password" autocapitalize="none" autocorrect="off" spellcheck="false">
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
    if current_user.is_authenticated and login_fresh():
        print(f"User already authenticated and fresh as: {current_user.username}, redirecting to index")
        return redirect('/index.html')
    elif current_user.is_authenticated and not login_fresh():
        # Remember-cookie 상태 등 비-프레시 세션이면 로그인 페이지를 보여서 재인증 유도
        print("User authenticated but session not fresh; showing login page for reauthentication")
    
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        print(f"Login attempt: username={username}")
        
        if username in users and users[username].password == password:
            # 로그인 성공 시 세션에 사용자 정보 저장
            user = users[username]
            login_user(user, remember=False)  # 2분 세션 만료를 위해 remember 비활성화
            session['user_id'] = user.id
            session['username'] = username
            session.permanent = True
            session.modified = True  # 세션 변경 사항 즉시 적용
            
            print(f"Login successful for user: {username}, ID: {user.id}")
            
            # 리디렉션 처리
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/') and next_page != '/login':
                print(f"Redirecting to: {next_page}")
                return redirect(next_page)
            print("Redirecting to index.html")
            return redirect(url_for('serve_index_html'))
        else:
            error = 'Invalid username or password'
            print(f"Login failed: {error}")
    
    return render_template_string(LOGIN_TEMPLATE, error=error)

@app.route('/logout')
def logout():
    logout_user()
    # Clear server-side session fully
    try:
        session.clear()
    except Exception as e:
        print(f"[DEBUG] Error clearing session on logout: {e}")
    # Ensure remember cookie is removed by setting an expired cookie
    resp = redirect(url_for('login'))
    try:
        resp.delete_cookie(
            key='remember_token',
            path='/',
            samesite='None',
            secure=True,
            httponly=True,
        )
    except Exception as e:
        print(f"[DEBUG] Error deleting remember_token cookie: {e}")
    return resp

# 정적 파일 서빙을 위한 라우트 (로그인 불필요)
@app.route('/static/<path:filename>')
def serve_static(filename):
    print(f"Serving static file: {filename}")
    resp = send_from_directory(app.static_folder, filename)
    # Prevent caching of static assets to reflect latest frontend changes
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

# 인덱스 HTML 직접 서빙 (로그인 필요)
@app.route('/index.html')
def serve_index_html():
    # 세션 및 쿠키 디버그 정보
    print(f"Request to /index.html - Session data: {dict(session)}")
    print(f"Request to /index.html - Cookies: {request.cookies}")
    print(f"Request to /index.html - User authenticated: {current_user.is_authenticated}")
    
    # 인증 확인 (fresh session only)
    if not current_user.is_authenticated or not login_fresh():
        print("User not authenticated, redirecting to login")
        return redirect(url_for('login'))
    
    print(f"Serving index.html for authenticated user: {current_user.username} (ID: {current_user.id})")
    # 세션 상태 디버그
    print(f"Session data: user_id={session.get('user_id')}, username={session.get('username')}, is_permanent={session.get('permanent', False)}")
    
    # 세션 만료를 의도대로 유지하기 위해 여기서 세션을 갱신하지 않습니다.
    # 주의: 세션에 쓰기(또는 session.modified=True)는 Flask-Session에서 만료시간을 연장할 수 있습니다.
    
    # index.html을 읽어 하트비트 스크립트를 주입
    index_path = os.path.join(app.static_folder, 'index.html')
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            html = f.read()
    except Exception as e:
        print(f"[DEBUG] Failed to read index.html for injection: {e}")
        return send_from_directory(app.static_folder, 'index.html')

    heartbeat_script = """
    <script>
    (function(){
      // 1) 세션 상태 주기 체크 (만료시 로그인으로)
      function checkSession(){
        fetch('/api/status', {credentials: 'include', redirect: 'manual'}).then(function(res){
          var redirected = res.redirected || (res.url && res.url.indexOf('/login') !== -1);
          if(res.status !== 200 || redirected){
            window.location.href = '/login';
          }
        }).catch(function(){
          // 네트워크 오류 등도 로그인으로 유도
          window.location.href = '/login';
        });
      }
      checkSession();
      setInterval(checkSession, 30000);

      // 2) 사용자 비활성(무동작) 2분 후 자동 로그아웃
      var idleMs = 120000; // 2분
      var idleTimer;
      function triggerLogout(){
        // 서버 세션 정리 후 로그인 화면으로
        window.location.href = '/logout';
      }
      function resetIdle(){
        if (idleTimer) clearTimeout(idleTimer);
        idleTimer = setTimeout(triggerLogout, idleMs);
      }
      ['click','mousemove','keydown','scroll','touchstart','visibilitychange'].forEach(function(evt){
        window.addEventListener(evt, resetIdle, {passive:true});
      });
      resetIdle();
    })();
    </script>
    """

    try:
        if '</body>' in html:
            html = html.replace('</body>', heartbeat_script + '</body>')
        else:
            html = html + heartbeat_script
    except Exception as e:
        print(f"[DEBUG] Failed to inject heartbeat script: {e}")
        return send_from_directory(app.static_folder, 'index.html')

    resp = make_response(html)
    # Prevent sensitive pages from being cached
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

# Static files should be accessible without login requirements
@app.route('/static/<path:filename>')
def static_files(filename):
    print(f"Serving static file: {filename}")
    # Two possible locations after CRA build copy:
    # 1) Top-level:    static/<filename>
    # 2) Nested build: static/static/<filename>
    top_level_path = os.path.join(app.static_folder, filename)
    nested_dir = os.path.join(app.static_folder, 'static')
    nested_path = os.path.join(nested_dir, filename)

    try:
        if os.path.exists(top_level_path):
            return send_from_directory(app.static_folder, filename)
        elif os.path.exists(nested_path):
            # Serve from nested build directory
            return send_from_directory(nested_dir, filename)
        else:
            # Fallback: try as-is (may help in some edge cases)
            return send_from_directory(app.static_folder, filename)
    except Exception as e:
        print(f"[DEBUG] Error serving static file '{filename}': {e}")
        # Final fallback to avoid leaking stack traces
        return ('Not Found', 404)

# Add explicit handlers for JS files that are failing
@app.route('/static/js/<path:filename>')
def static_js_files(filename):
    print(f"Serving JS file: {filename}")
    # Try top-level static/js and nested static/static/js
    top_js_dir = os.path.join(app.static_folder, 'js')
    nested_js_dir = os.path.join(app.static_folder, 'static', 'js')
    top_js_path = os.path.join(top_js_dir, filename)
    nested_js_path = os.path.join(nested_js_dir, filename)

    try:
        if os.path.exists(top_js_path):
            return send_from_directory(top_js_dir, filename)
        elif os.path.exists(nested_js_path):
            return send_from_directory(nested_js_dir, filename)
        else:
            # As a fallback, let the generic static handler try
            return static_files(os.path.join('js', filename))
    except Exception as e:
        print(f"[DEBUG] Error serving JS file '{filename}': {e}")
        return ('Not Found', 404)

# 기본 경로 및 기타 경로 처리 (로그인 필요)
@app.route('/', defaults={'path': ''}, methods=['GET'])
@app.route('/<path:path>', methods=['GET'])
@fresh_login_required
def serve_react(path):
    """Serve React frontend"""
    print(f"Serving React frontend for path: {path}, user: {current_user.username if current_user.is_authenticated else 'not authenticated'}")
    # 정적 파일 처리는 이제 별도 라우트에서 처리
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        resp = send_from_directory(app.static_folder, path)
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp
    else:
        # React 앱의 index.html 서빙 (하트비트 스크립트 주입)
        index_path = os.path.join(app.static_folder, 'index.html')
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                html = f.read()
        except Exception as e:
            print(f"[DEBUG] Failed to read index.html for injection (serve_react): {e}")
            resp = send_from_directory(app.static_folder, 'index.html')
            resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            resp.headers['Pragma'] = 'no-cache'
            resp.headers['Expires'] = '0'
            return resp

        heartbeat_script = """
        <script>
        (function(){
          // 1) 세션 상태 주기 체크 (만료시 로그인으로)
          function checkSession(){
            fetch('/api/status', {credentials: 'include', redirect: 'manual'}).then(function(res){
              var redirected = res.redirected || (res.url && res.url.indexOf('/login') !== -1);
              if(res.status !== 200 || redirected){
                window.location.href = '/login';
              }
            }).catch(function(){
              window.location.href = '/login';
            });
          }
          checkSession();
          setInterval(checkSession, 30000);

          // 2) 사용자 비활성(무동작) 2분 후 자동 로그아웃
          var idleMs = 120000; // 2분
          var idleTimer;
          function triggerLogout(){
            window.location.href = '/logout';
          }
          function resetIdle(){
            if (idleTimer) clearTimeout(idleTimer);
            idleTimer = setTimeout(triggerLogout, idleMs);
          }
          ['click','mousemove','keydown','scroll','touchstart','visibilitychange'].forEach(function(evt){
            window.addEventListener(evt, resetIdle, {passive:true});
          });
          resetIdle();
        })();
        </script>
        """

        try:
            if '</body>' in html:
                html = html.replace('</body>', heartbeat_script + '</body>')
            else:
                html = html + heartbeat_script
        except Exception as e:
            print(f"[DEBUG] Failed to inject heartbeat script (serve_react): {e}")
            resp = send_from_directory(app.static_folder, 'index.html')
            resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            resp.headers['Pragma'] = 'no-cache'
            resp.headers['Expires'] = '0'
            return resp

        resp = make_response(html)
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp

@app.route('/similar-images', methods=['GET'])
@fresh_login_required
def similar_images_page():
    """Serve similar images search page"""
    resp = send_from_directory(app.static_folder, 'similar-images.html')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/object-detection-search', methods=['GET'])
@fresh_login_required
def object_detection_search_page():
    """Serve object detection search page"""
    resp = send_from_directory(app.static_folder, 'object-detection-search.html')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/model-vector-db', methods=['GET'])
@fresh_login_required
def model_vector_db_page():
    """Serve model vector DB UI page"""
    resp = send_from_directory(app.static_folder, 'model-vector-db.html')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/openai-chat', methods=['GET'])
@fresh_login_required
def openai_chat_page():
    """Serve OpenAI chat UI page"""
    resp = send_from_directory(app.static_folder, 'openai-chat.html')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/api/openai/chat', methods=['POST'])
@fresh_login_required
def openai_chat_api():
    """Forward chat request to OpenAI Chat Completions API.
    Expects JSON: { prompt: string, model?: string, api_key?: string, system?: string }
    Uses OPENAI_API_KEY from environment if api_key not provided.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    prompt = (data or {}).get('prompt', '').strip()
    model = (data or {}).get('model') or os.environ.get('OPENAI_MODEL', 'gpt-4')
    system = (data or {}).get('system') or 'You are a helpful assistant.'
    api_key = (data or {}).get('api_key') or os.environ.get('OPENAI_API_KEY')

    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400
    if not api_key:
        return jsonify({"error": "Missing OpenAI API key. Provide in request or set OPENAI_API_KEY env."}), 400

    # Prefer official Python SDK if available
    if OpenAI is None:
        return jsonify({"error": "OpenAI Python package not installed on server"}), 500

    try:
        start = time.time()
        client = OpenAI(api_key=api_key)

        # Perform chat completion
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        latency = round(time.time() - start, 3)
    except Exception as e:
        # Log detailed error for diagnostics
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass
        err_msg = str(e)
        print(f"[OpenAI Chat Error] model={model} err={err_msg}")
        return jsonify({
            "error": "OpenAI SDK call failed",
            "detail": err_msg,
            "model": model
        }), 502

    try:
        content = chat.choices[0].message.content if chat and chat.choices else ''
        usage = getattr(chat, 'usage', None)
        usage = usage.model_dump() if hasattr(usage, 'model_dump') else (usage or {})
    except Exception as e:
        return jsonify({"error": f"Failed to parse SDK response: {str(e)}"}), 500

    return jsonify({
        'response': content,
        'model': model,
        'usage': usage,
        'latency_sec': latency
    })

@app.route('/api/vision-rag/query', methods=['POST'])
@login_required
def vision_rag_query():
    """Vision RAG endpoint.
    Expects JSON with one of the following query modes and a user question:
      - { userQuery, searchType: 'image', image, n_results? }
      - { userQuery, searchType: 'object', objectId, n_results? }
      - { userQuery, searchType: 'class', class_name, n_results? }
    Returns: { answer, retrieved: [...], model, latency_sec }
    """
    if ChatOpenAI is None:
        return jsonify({"error": "LangChain not installed on server"}), 500

    data = request.get_json(silent=True) or {}
    # Debug: log incoming payload keys and basic info (without sensitive data)
    try:
        print("[VRAG][REQ] keys=", list(data.keys()))
        print("[VRAG][REQ] has_api_key=", bool(data.get('api_key') or os.environ.get('OPENAI_API_KEY')))
        _img = data.get('image')
        if isinstance(_img, str):
            print("[VRAG][REQ] image_str_len=", len(_img), "prefix=", _img[:30] if len(_img) > 30 else _img)
        print("[VRAG][REQ] searchType=", data.get('searchType'), "objectId=", data.get('objectId'), "class_name=", data.get('class_name'))
    except Exception as _e:
        print("[VRAG][WARN] failed to log request payload:", _e)

    user_query = (data.get('userQuery') or '').strip()
    if not user_query:
        return jsonify({"error": "Missing 'userQuery'"}), 400

    api_key = data.get('api_key') or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return jsonify({"error": "Missing OpenAI API key. Provide in request or set OPENAI_API_KEY env."}), 400

    search_type = data.get('searchType', 'image')
    n_results = int(data.get('n_results', 5))
    print(f"[VRAG] user_query='{user_query}' | search_type={search_type} | n_results={n_results}")

    # Build query embedding or filtered fetch similar to /api/search-similar-objects
    results = None
    try:
        if search_type == 'image' and 'image' in data:
            image_data = data['image']
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
            query_embedding = generate_image_embedding(image)
            if query_embedding is None:
                return jsonify({"error": "Failed to generate image embedding"}), 500
            results = object_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "distances"]
            ) if object_collection is not None else None
        elif search_type == 'object' and 'objectId' in data:
            obj_id = data['objectId']
            base = object_collection.get(ids=[obj_id], include=["embeddings"]) if object_collection is not None else None
            emb = base["embeddings"][0] if base and "embeddings" in base and base["embeddings"] else None
            if emb is None:
                return jsonify({"error": "objectId not found or has no embedding"}), 400
            results = object_collection.query(
                query_embeddings=[emb],
                n_results=n_results,
                include=["metadatas", "distances"]
            )
        elif search_type == 'class' and 'class_name' in data:
            filter_query = {"class": {"$eq": data['class_name']}}
            results = object_collection.get(
                where=filter_query,
                limit=n_results,
                include=["metadatas", "embeddings", "documents"]
            ) if object_collection is not None else None
        else:
            return jsonify({"error": "Invalid search parameters"}), 400
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    # Format results using existing helper
    formatted = format_object_results(results) if results else []
    # Debug: log retrieval summary
    try:
        cnt = len(formatted)
        print(f"[VRAG][RETRIEVE] items={cnt}")
        if cnt:
            print("[VRAG][RETRIEVE] first_item=", {
                'id': formatted[0].get('id'),
                'distance': formatted[0].get('distance'),
                'meta_keys': list((formatted[0].get('metadata') or {}).keys())
            })
    except Exception as _e:
        print("[VRAG][WARN] failed to log retrieval summary:", _e)

    # Build concise context for LLM
    def _shorten(md):
        try:
            bbox = md.get('bbox') if isinstance(md, dict) else None
            if isinstance(bbox, dict):
                bbox = {k: round(float(v), 3) for k, v in bbox.items() if isinstance(v, (int, float))}
            return {
                'image_id': md.get('image_id'),
                'class': md.get('class'),
                'confidence': md.get('confidence'),
                'bbox': bbox,
            }
        except Exception:
            return {k: md.get(k) for k in ('image_id', 'class', 'confidence') if k in md}

    context_items = []
    for r in formatted[:n_results]:
        md = r.get('metadata', {})
        item = {
            'id': r.get('id'),
            'distance': r.get('distance'),
            'meta': _shorten(md)
        }
        context_items.append(item)

    # Compose prompt
    system_text = (
        "You are a vision assistant. Use ONLY the provided detected object context to answer. "
        "Be concise and state uncertainty if context is insufficient."
    )
    # Provide the minimal JSON-like context to the model
    context_text = json.dumps(context_items, ensure_ascii=False, indent=2)
    user_text = f"User question: {user_query}\n\nDetected context (top {len(context_items)}):\n{context_text}"

    # Debug: show compact context preview
    try:
        print("[VRAG][CTX] context_items_count=", len(context_items))
        if context_items:
            print("[VRAG][CTX] sample=", json.dumps(context_items[0], ensure_ascii=False))
    except Exception as _e:
        print("[VRAG][WARN] failed to log context:", _e)

    # Attempt multimodal call (text + top-1 image) if available; otherwise fallback to text-only LangChain.
    answer = None
    model_used = None
    try:
        start = time.time()
        top_data_url = None
        try:
            if formatted:
                md0 = (formatted[0] or {}).get('metadata') or {}
                img_b64 = md0.get('image_data')
                if isinstance(img_b64, str) and len(img_b64) > 50:
                    # Construct data URL without logging raw base64
                    top_data_url = 'data:image/jpeg;base64,' + img_b64
        except Exception:
            top_data_url = None

        # Prefer OpenAI SDK for multimodal if available and we have an image
        if OpenAI is not None and top_data_url is not None:
            client = OpenAI(api_key=api_key)
            model_used = os.environ.get('OPENAI_MODEL', 'gpt-4o')
            chat = client.chat.completions.create(
                model=model_used,
                messages=[
                    {"role": "system", "content": system_text},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": top_data_url}},
                        ],
                    },
                ],
            )
            answer = chat.choices[0].message.content if chat and chat.choices else ''
        else:
            # Fallback to existing LangChain text-only flow
            llm = ChatOpenAI(api_key=api_key, model=os.environ.get('OPENAI_MODEL', 'gpt-4o'))
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_text),
                ("human", "{input}")
            ])
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"input": user_text})
            model_used = getattr(llm, 'model', None)

        latency = round(time.time() - start, 3)
    except Exception as e:
        return jsonify({"error": f"LLM call failed: {str(e)}"}), 502

    return jsonify({
        "answer": answer,
        "retrieved": context_items,
        "model": model_used,
        "latency_sec": latency
    })

@app.route('/api/status', methods=['GET'])
@fresh_login_required
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
