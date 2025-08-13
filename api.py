# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_from_directory
import torch
from PIL import Image
import numpy as np
import os
import io
from io import BytesIO
import base64
import uuid
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from flask_cors import CORS
import json

# Fix for SQLite3 version compatibility with ChromaDB
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3

import chromadb
from chromadb.utils import embedding_functions

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

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
        print("Downloading YOLOv8 model to {}".format(model_path))
        try:
            os.system("wget -q https://ultralytics.com/assets/yolov8n.pt -O {}".format(model_path))
            print("YOLOv8 model downloaded successfully")
        except Exception as e:
            print("Error downloading YOLOv8 model: {}".format(e))
            # 다운로드 실패 시 대체 URL 시도
            try:
                os.system("wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O {}".format(model_path))
                print("YOLOv8 model downloaded from alternative source")
            except Exception as e2:
                print("Error downloading from alternative source: {}".format(e2))
                # 마지막 대안으로 직접 모델 URL 사용
                try:
                    os.system("curl -L https://ultralytics.com/assets/yolov8n.pt --output {}".format(model_path))
                    print("YOLOv8 model downloaded using curl")
                except Exception as e3:
                    print("All download attempts failed: {}".format(e3))
    
    # 환경 변수 설정 - 설정 파일 경로 지정
    os.environ["YOLO_CONFIG_DIR"] = temp_dir
    os.environ["MPLCONFIGDIR"] = temp_dir
    
    yolo_model = YOLO(model_path)  # Using the nano model for faster inference
    print("YOLOv8 model loaded successfully")
except Exception as e:
    print("Error loading YOLOv8 model:", e)
    yolo_model = None

# DETR model (DEtection TRansformer)
detr_model = None
detr_processor = None
try:
    from transformers import DetrForObjectDetection, DetrImageProcessor
    
    # 임시 디렉토리 사용
    import tempfile
    temp_dir = tempfile.gettempdir()
    os.environ["TRANSFORMERS_CACHE"] = temp_dir
    
    detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    print("DETR model loaded successfully")
except Exception as e:
    print("Error loading DETR model:", e)
    detr_model = None
    detr_processor = None

# ViT model
vit_model = None
vit_processor = None
try:
    from transformers import ViTForImageClassification, ViTImageProcessor
    
    # 임시 디렉토리 사용
    import tempfile
    temp_dir = tempfile.gettempdir()
    os.environ["TRANSFORMERS_CACHE"] = temp_dir
    
    vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    print("ViT model loaded successfully")
except Exception as e:
    print("Error loading ViT model:", e)
    vit_model = None
    vit_processor = None

# LLM 모델 초기화 (TinyLlama 사용)
llm_model = None
llm_tokenizer = None
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 임시 디렉토리 사용
    import tempfile
    temp_dir = tempfile.gettempdir()
    os.environ["TRANSFORMERS_CACHE"] = temp_dir
    
    # TinyLlama 모델 로드 (오픈 액세스 모델)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    print("LLM model loaded successfully")
except Exception as e:
    print("Error loading LLM model:", e)
    llm_model = None
    llm_tokenizer = None

# Get device information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

def format_vision_results(vision_results):
    """Format vision results into a text description"""
    description = ""
    for result in vision_results:
        description += "{} (confidence: {})\n".format(result['class'], result['confidence'])
    return description

def process_llm_query(vision_results, user_query):
    """Process a query with the LLM model using vision results and user text"""
    if llm_model is None or llm_tokenizer is None:
        return {"error": "LLM model not available"}
    
    try:
        # Format the vision results into a text description
        vision_description = format_vision_results(vision_results)
        
        # Create a prompt for the LLM
        prompt = """You are a helpful AI assistant that can analyze images. 
        I'll provide you with the results from computer vision models that have analyzed an image, and you'll help interpret these results.
        
        Here's what the vision models detected in the image:
        {}
        
        User query: {}
        
        Please provide a helpful response to the user's query based on the vision results.""".format(vision_description, user_query)
        
        # 토큰 길이 확인 및 조정
        tokens = llm_tokenizer.encode(prompt)
        if len(tokens) > 1500:  # 토큰 수가 너무 많은 경우 요약
            prompt = """You are a helpful AI assistant that can analyze images. 
            I'll provide you with a summary of what vision models detected in an image, and you'll help interpret these results.
            
            Summary of detections: {} objects were detected.
            
            User query: {}
            
            Please provide a helpful response to the user's query based on the vision results.""".format(len(vision_results), user_query)
        
        # Generate response with the LLM
        inputs = llm_tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = llm_model.generate(
                inputs["input_ids"],
                max_length=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
        
        response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response (removing the prompt)
        response = response.split("User query:")[-1].split(user_query)[-1].strip()
        
        return {"response": response}
    
    except Exception as e:
        return {"error": "Error processing LLM query: {}".format(str(e))}

def image_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def process_yolo(image):
    """Process an image with YOLOv8 model"""
    if yolo_model is None:
        return {"error": "YOLOv8 model not available"}
    
    try:
        start_time = time.time()
        
        # Convert image if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Run inference
        results = yolo_model(image)
        
        # Process results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].tolist()  # get box coordinates in (x1, y1, x2, y2) format
                c = box.cls.item()
                conf = box.conf.item()
                class_name = r.names[c]
                
                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 3),
                    "bbox": [round(x, 2) for x in b]
                })
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Add inference time and device info
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        
        return {
            "detections": detections,
            "performance": {
                "inference_time": round(inference_time, 3),
                "device": device_info
            }
        }
    
    except Exception as e:
        print("Error in YOLO processing: {}".format(e))
        return {"error": str(e)}

def process_detr(image):
    """Process an image with DETR model"""
    if detr_model is None or detr_processor is None:
        return {"error": "DETR model not available"}
    
    try:
        start_time = time.time()
        
        # Prepare image for the model
        inputs = detr_processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            detr_model.to("cuda")
        
        # Run inference
        with torch.no_grad():
            outputs = detr_model(**inputs)
        
        # Convert outputs to COCO API format
        target_sizes = torch.tensor([image.size[::-1]])
        results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
        
        # Process results
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.tolist()
            class_name = detr_model.config.id2label[label.item()]
            detections.append({
                "class": class_name,
                "confidence": round(score.item(), 3),
                "bbox": [round(x, 2) for x in box]
            })
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Add inference time and device info
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        
        return {
            "detections": detections,
            "performance": {
                "inference_time": round(inference_time, 3),
                "device": device_info
            }
        }
    
    except Exception as e:
        print("Error in DETR processing: {}".format(e))
        return {"error": str(e)}

def process_vit(image):
    """Process an image with ViT model"""
    if vit_model is None or vit_processor is None:
        return {"error": "ViT model not available"}
    
    try:
        start_time = time.time()
        
        # Prepare image for the model
        inputs = vit_processor(images=image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            vit_model.to("cuda")
        
        # Run inference
        with torch.no_grad():
            outputs = vit_model(**inputs)
        
        # Process results
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
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
    
    except Exception as e:
        print("Error in ViT processing: {}".format(e))
        return {"error": str(e)}

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

@app.route('/api/analyze', methods=['POST'])
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
        print("Error generating image embedding: {}".format(e))
        return None

@app.route('/api/similar-images', methods=['POST'])
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
        print("Error in similar-images API: {}".format(e))
        return jsonify({"error": str(e)}), 500

@app.route('/api/add-to-collection', methods=['POST'])
def add_to_collection():
    """이미지를 벡터 DB에 추가하는 API"""
    if clip_model is None or clip_processor is None or image_collection is None:
        return jsonify({"error": "Image embedding model or vector DB not available"})

    try:
        # JSON 또는 form-data 모두 지원
        metadata = {}
        image_id = None
        image = None
        image_base64 = None

        if request.is_json:
            data = request.get_json(silent=True) or {}
            if 'image' not in data:
                return jsonify({"error": "No image provided"})
            image_id = data.get('id', str(uuid.uuid4()))
            metadata = data.get('metadata', {}) or {}
            image_data = data['image']
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
            image_base64 = image_data
        else:
            # 요청에서 이미지 데이터 추출 (form)
            if 'image' not in request.files and 'image' not in request.form:
                return jsonify({"error": "No image provided"})

            # 메타데이터 추출
            if 'metadata' in request.form:
                metadata = json.loads(request.form['metadata'])

            # 이미지 ID (제공되지 않은 경우 자동 생성)
            image_id = request.form.get('id', str(uuid.uuid4()))

            if 'image' in request.files:
                # 파일로 업로드된 경우
                image_file = request.files['image']
                image = Image.open(image_file).convert('RGB')
                # 이미지를 base64로 인코딩하여 메타데이터에 저장
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            else:
                # base64로 인코딩된 경우
                image_data = request.form['image']
                if image_data.startswith('data:image'):
                    # Remove the data URL prefix if present
                    image_data = image_data.split(',')[1]
                image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
                image_base64 = image_data  # 이미 base64 형식

        # 이미지 데이터를 메타데이터에 추가
        metadata['image_data'] = image_base64

        # 이미지 임베딩 생성
        embedding = generate_image_embedding(image)
        if embedding is None:
            return jsonify({"error": "Failed to generate image embedding"})

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
        print("Error in add-to-collection API: {}".format(e))
        return jsonify({"error": str(e)}), 500

@app.route('/api/add-detected-objects', methods=['POST'])
def add_detected_objects():
    """객체 인식 결과를 벡터 DB에 추가하는 API (JSON 지원)"""
    if clip_model is None or object_collection is None:
        return jsonify({"error": "Image embedding model or vector DB not available"})

    try:
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({"error": "No data received"})

        if 'image' not in data or 'objects' not in data:
            return jsonify({"error": "Missing image or objects data"})

        image_data = data['image']
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
        image_width, image_height = image.size

        image_id = data.get('imageId', str(uuid.uuid4()))

        object_ids = []
        object_embeddings = []
        object_metadatas = []

        for obj in data.get('objects', []):
            object_id = f"{image_id}_{str(uuid.uuid4())[:8]}"

            bbox = obj.get('bbox', [])
            # bbox could be [x1,y1,x2,y2] in absolute pixels from frontend; normalize to [0-1]
            if isinstance(bbox, list) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                nx1 = x1 / image_width
                ny1 = y1 / image_height
                nx2 = x2 / image_width
                ny2 = y2 / image_height
                nwidth = nx2 - nx1
                nheight = ny2 - ny1
            elif isinstance(bbox, dict):
                nx1 = bbox.get('x', 0)
                ny1 = bbox.get('y', 0)
                nwidth = bbox.get('width', 0)
                nheight = bbox.get('height', 0)
            else:
                nx1 = ny1 = nwidth = nheight = 0

            # crop using absolute pixels
            x1_px = int(nx1 * image_width)
            y1_px = int(ny1 * image_height)
            w_px = int(nwidth * image_width)
            h_px = int(nheight * image_height)
            try:
                object_image = image.crop((x1_px, y1_px, x1_px + w_px, y1_px + h_px))
            except Exception:
                # fallback: if bbox was absolute already
                try:
                    object_image = image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                except Exception:
                    continue

            embedding = generate_image_embedding(object_image)
            if embedding is None:
                continue

            buffered = BytesIO()
            object_image.save(buffered, format="JPEG")
            obj_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            bbox_json = json.dumps({
                "x": nx1, "y": ny1, "width": nwidth, "height": nheight
            })

            metadata = {
                "image_id": image_id,
                "class": obj.get('class', ''),
                "confidence": obj.get('confidence', 0),
                "bbox": bbox_json,
                "image_data": obj_b64
            }

            object_ids.append(object_id)
            object_embeddings.append(embedding)
            object_metadatas.append(metadata)

        if not object_ids:
            return jsonify({"error": "No valid objects to add"})

        object_collection.add(
            ids=object_ids,
            embeddings=object_embeddings,
            metadatas=object_metadatas
        )

        return jsonify({
            "success": True,
            "image_id": image_id,
            "object_count": len(object_ids),
            "object_ids": object_ids
        })
    except Exception as e:
        print("Error in add-detected-objects API: {}".format(e))
        return jsonify({"error": str(e)}), 500

@app.route('/api/search-similar-objects', methods=['POST'])
def search_similar_objects():
    """유사한 객체 검색 API (JSON)"""
    if clip_model is None or object_collection is None:
        return jsonify({"error": "Image embedding model or vector DB not available"})

    try:
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({"error": "Missing request data"})

        search_type = data.get('searchType', 'image')
        n_results = int(data.get('n_results', 5))

        query_embedding = None

        if search_type == 'image' and 'image' in data:
            image_data = data['image']
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
            query_embedding = generate_image_embedding(image)
        elif search_type == 'object' and 'objectId' in data:
            result = object_collection.get(ids=[data['objectId']], include=["embeddings"])
            if result and "embeddings" in result and len(result["embeddings"]) > 0:
                query_embedding = result["embeddings"][0]
        elif search_type == 'class' and 'class_name' in data:
            results = object_collection.query(
                query_embeddings=None,
                where={"class": {"$eq": data['class_name']}},
                n_results=n_results,
                include=["metadatas", "distances"]
            )
            formatted = format_object_results(results)
            return jsonify({"success": True, "searchType": "class", "results": formatted})
        else:
            return jsonify({"error": "Invalid search parameters"})

        if query_embedding is None:
            return jsonify({"error": "Failed to generate query embedding"})

        results = object_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "distances"]
        )
        formatted = format_object_results(results)
        return jsonify({"success": True, "searchType": search_type, "results": formatted})
    except Exception as e:
        print("Error in search-similar-objects API: {}".format(e))
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "online",
        "models": {
            "yolo": yolo_model is not None,
            "detr": detr_model is not None and detr_processor is not None,
            "vit": vit_model is not None and vit_processor is not None,
            "clip": clip_model is not None and clip_processor is not None,
            "llm": llm_model is not None and llm_tokenizer is not None
        },
        "vector_db": {
            "available": vector_db is not None,
            "image_collection": image_collection is not None,
            "object_collection": object_collection is not None
        },
        "device": "GPU" if torch.cuda.is_available() else "CPU"
    })

@app.route('/api/add-object', methods=['POST'])
def add_object_to_collection():
    """Detected object를 벡터 DB에 추가하는 API"""
    if clip_model is None or clip_processor is None or object_collection is None:
        return jsonify({"error": "Object embedding model or vector DB not available"})

    try:
        # 요청에서 이미지 데이터 추출
        if 'image' not in request.files and 'image' not in request.form:
            return jsonify({"error": "No image provided"})

        # 객체 정보 추출
        if 'bbox' not in request.form or 'class' not in request.form:
            return jsonify({"error": "Missing object information (bbox, class)"})

        # 바운딩 박스 및 클래스 정보 파싱
        bbox = json.loads(request.form['bbox'])
        class_name = request.form['class']
        confidence = float(request.form.get('confidence', 0.0))
        object_id = request.form.get('id', str(uuid.uuid4()))

        # 메타데이터 추출
        metadata = {
            "class": class_name,
            "confidence": confidence,
            "bbox": json.dumps(bbox)  # ChromaDB는 중첩 객체를 지원하지 않으므로 JSON 문자열로 저장
        }

        if 'parent_image_id' in request.form:
            metadata["parent_image_id"] = request.form['parent_image_id']

        if 'image' in request.files:
            # 파일로 업로드된 경우
            image_file = request.files['image']
            full_image = Image.open(image_file).convert('RGB')
            # 원본 이미지 데이터를 base64로 변환
            image_data_raw = io.BytesIO()
            full_image.save(image_data_raw, format="JPEG")
            image_data = base64.b64encode(image_data_raw.getvalue()).decode('utf-8')
        else:
            # base64로 인코딩된 경우
            image_data = request.form['image']
            if image_data.startswith('data:image'):
                # Remove the data URL prefix if present
                image_data = image_data.split(',')[1]
            full_image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')

        # 바운딩 박스에서 객체 이미지 추출
        object_image = full_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        # 객체 이미지를 base64로 변환하여 메타데이터에 추가
        buffered = io.BytesIO()
        object_image.save(buffered, format="JPEG")
        object_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        metadata['image_data'] = object_image_base64

        # 객체 이미지 임베딩 생성
        embedding = generate_image_embedding(object_image)
        if embedding is None:
            return jsonify({"error": "Failed to generate object embedding"})

        # 객체를 DB에 추가
        object_collection.add(
            ids=[object_id],
            embeddings=[embedding],
            metadatas=[metadata]
        )

        return jsonify({
            "success": True,
            "object_id": object_id,
            "message": "Object added to collection"
        })

    except Exception as e:
        print("Error in add-object API: {}".format(e))
        return jsonify({"error": str(e)}), 500

@app.route('/api/search-objects', methods=['POST'])
def search_objects():
    """객체 검색 API"""
    if object_collection is None:
        return jsonify({"error": "Object collection not available"})

    try:
        # 검색 방법 결정
        search_by = request.form.get('search_by', 'image')

        if search_by == 'image':
            # 이미지로 검색
            if 'image' not in request.files and 'image' not in request.form:
                return jsonify({"error": "No image provided"})

            if 'image' in request.files:
                image_file = request.files['image']
                image = Image.open(image_file).convert('RGB')
            else:
                image_data = request.form['image']
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')

            # 이미지 임베딩 생성
            embedding = generate_image_embedding(image)
            if embedding is None:
                return jsonify({"error": "Failed to generate image embedding"})

            # 유사 객체 검색
            results = object_collection.query(
                query_embeddings=[embedding],
                n_results=10
            )

        elif search_by == 'id':
            # ID로 검색
            if 'id' not in request.form:
                return jsonify({"error": "No object ID provided"})

            object_id = request.form['id']
            results = object_collection.get(ids=[object_id])

        elif search_by == 'class':
            # 클래스로 검색
            if 'class' not in request.form:
                return jsonify({"error": "No class name provided"})

            class_name = request.form['class']
            results = object_collection.get(
                where={"class": class_name}
            )

        else:
            return jsonify({"error": "Invalid search method"}), 400

        # 결과 포맷팅
        objects = format_search_results(results)

        return jsonify({
            "objects": objects
        })

    except Exception as e:
        print("Error in search-objects API: {}".format(e))
        return jsonify({"error": str(e)}), 500

def format_search_results(results):
    """검색 결과를 일관된 형식으로 포맷팅"""
    formatted_results = []

    # ID 기반 또는 클래스 기반 검색 결과 처리
    if 'ids' in results and isinstance(results['ids'], list):
        for i, obj_id in enumerate(results['ids']):
            item = {
                "id": obj_id,
                "metadata": results['metadatas'][i] if 'metadatas' in results else {}
            }
            
            # 바운딩 박스가 JSON 문자열로 저장된 경우 파싱
            if 'metadata' in item and 'bbox' in item['metadata']:
                try:
                    item['metadata']['bbox'] = json.loads(item['metadata']['bbox'])
                except:
                    pass  # 파싱 실패 시 원본 유지
                    
            formatted_results.append(item)
    
    # 임베딩 기반 검색 결과 처리
    elif 'ids' in results and isinstance(results['ids'], list) and len(results['ids']) > 0:
        for i, obj_id in enumerate(results['ids'][0]):
            item = {
                "id": obj_id,
                "distance": float(results['distances'][0][i]) if 'distances' in results else 0.0,
                "metadata": results['metadatas'][0][i] if 'metadatas' in results else {}
            }
            
            # 바운딩 박스가 JSON 문자열로 저장된 경우 파싱
            if 'metadata' in item and 'bbox' in item['metadata']:
                try:
                    item['metadata']['bbox'] = json.loads(item['metadata']['bbox'])
                except:
                    pass  # 파싱 실패 시 원본 유지
                    
            formatted_results.append(item)
    
    return formatted_results

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path and os.path.exists(os.path.join('static', path)):
        return send_from_directory('static', path)
    return send_from_directory('static', 'index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host='0.0.0.0', port=port)
