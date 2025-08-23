"""
Product Comparison Multi-Agent System

This module implements a multi-agent system for product comparison based on images.
The system uses various specialized agents to process images, extract features,
compare products, and provide recommendations.

Main components:
- Coordinator: Orchestrates the multi-agent workflow
- ImageProcessingAgent: Handles image recognition and object detection
- FeatureExtractionAgent: Extracts product specifications and features
- ComparisonAgent: Compares products based on extracted features
- RecommendationAgent: Provides purchase recommendations

Each agent utilizes vision models and LLMs to perform its specialized tasks.
"""

import os
import uuid
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import base64
from io import BytesIO
from datetime import datetime
from threading import Thread

import torch
from PIL import Image
import numpy as np

# Import LangChain components for agent implementation
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from langchain.agents import create_openai_functions_agent
    from langchain.agents import AgentExecutor
    from langchain.memory import ConversationBufferMemory
    from langchain.tools.render import format_tool_to_openai_function
    from langchain_openai import ChatOpenAI
    from langchain_experimental.tools.python.tool import PythonAstREPLTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Warning: LangChain components not available. Product comparison will work with limited functionality.")
    # Set all LangChain components to None
    ChatPromptTemplate = None
    StrOutputParser = None
    JsonOutputParser = None
    create_openai_functions_agent = None
    AgentExecutor = None
    ConversationBufferMemory = None
    format_tool_to_openai_function = None
    ChatOpenAI = None
    PythonAstREPLTool = None
    LANGCHAIN_AVAILABLE = False

# Import vision models if available
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from transformers import CLIPProcessor, CLIPModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import ViTImageProcessor, ViTForImageClassification
except ImportError:
    CLIPProcessor = None
    CLIPModel = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    ViTImageProcessor = None
    ViTForImageClassification = None

# Session storage for SSE communication
class SessionManager:
    """Manages active product comparison sessions and their event streams"""
    
    def __init__(self):
        self.active_sessions = {}
        self.results = {}
        
    def create_session(self, session_id=None):
        """Create a new session with unique ID"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.active_sessions[session_id] = {
            "created_at": datetime.now(),
            "messages": [],
            "status": "created"
        }
        return session_id
    
    def add_message(self, session_id, message, agent_type="system"):
        """Add a message to the session event stream"""
        if session_id not in self.active_sessions:
            return False
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.active_sessions[session_id]["messages"].append({
            "message": message,
            "agent": agent_type,
            "timestamp": timestamp
        })
        return True
    
    def get_messages(self, session_id):
        """Get all messages for a session"""
        if session_id not in self.active_sessions:
            return []
        return self.active_sessions[session_id]["messages"]
    
    def set_final_result(self, session_id, result):
        """Store the final analysis result for a session"""
        self.results[session_id] = result
        self.active_sessions[session_id]["status"] = "completed"
    
    def get_final_result(self, session_id):
        """Get the final result for a session"""
        return self.results.get(session_id)
    
    def set_status(self, session_id, status):
        """Update session status"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = status
    
    def get_status(self, session_id):
        """Get session status"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]["status"]
        return None
    
    def clean_old_sessions(self, max_age_hours=24):
        """Clean up old sessions"""
        now = datetime.now()
        sessions_to_remove = []
        
        for session_id, session_data in self.active_sessions.items():
            age = now - session_data["created_at"]
            if age.total_seconds() > max_age_hours * 3600:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
            if session_id in self.results:
                del self.results[session_id]

# Initialize the session manager
session_manager = SessionManager()

# Base Agent Class
class BaseAgent:
    """Base class for all specialized agents"""
    
    def __init__(self, name, llm=None):
        self.name = name
        
        # Use LangChain ChatOpenAI as the default LLM if none is provided
        if llm is None:
            try:
                if LANGCHAIN_AVAILABLE and ChatOpenAI is not None:
                    # Initialize ChatOpenAI with environment variable for API key
                    api_key = os.environ.get('OPENAI_API_KEY')
                    if api_key:
                        self.llm = ChatOpenAI(
                            model="gpt-5-mini",
                            temperature=0.7,
                            api_key=api_key
                        )
                        print(f"Initialized {name} with ChatOpenAI (gpt-4)")
                    else:
                        print(f"Warning: OPENAI_API_KEY not found. {name} will use fallback mode.")
                        self.llm = None
                else:
                    print(f"Warning: LangChain not available. {name} will use fallback mode.")
                    self.llm = None
            except Exception as e:
                print(f"Error initializing ChatOpenAI for {name}: {e}")
                self.llm = None
        else:
            self.llm = llm
    
    def log(self, session_id, message):
        """Log a message to the session"""
        return session_manager.add_message(session_id, message, agent_type=self.name)
    
    async def process(self, session_id, data):
        """Process data with this agent - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")


class ImageProcessingAgent(BaseAgent):
    """Agent responsible for image processing and object recognition
    
    This agent uses computer vision models to detect product type, brand, model,
    and other visual characteristics from product images.
    """
    
    def __init__(self, name="ImageProcessingAgent"):
        super().__init__(name)
        # Initialize vision models
        self.models = self._load_vision_models()
    
    def _load_vision_models(self):
        """Load vision models for product recognition"""
        models = {}
        
        # Try to load YOLO for object detection
        try:
            if YOLO is not None:
                models["yolo"] = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"Error loading YOLO: {e}")
        
        # Try to load ViT for image classification
        try:
            if ViTImageProcessor is not None and ViTForImageClassification is not None:
                models["vit_processor"] = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
                models["vit_model"] = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        except Exception as e:
            print(f"Error loading ViT: {e}")
        
        # Try to load CLIP for visual embedding
        try:
            if CLIPProcessor is not None and CLIPModel is not None:
                models["clip_processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                models["clip_model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as e:
            print(f"Error loading CLIP: {e}")
            
        return models
    
    def _process_with_yolo(self, image):
        """Process image with YOLO for object detection"""
        if "yolo" not in self.models:
            return {"error": "YOLO model not available"}
            
        # Convert image to numpy array if it's a PIL image
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Run inference
        results = self.models["yolo"](image_np)
        
        # Extract detection results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                confidence = round(box.conf[0].item(), 2)
                bbox = box.xyxy[0].tolist()
                bbox = [round(x) for x in bbox]
                
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": bbox
                })
        
        return {"detections": detections}
    
    def _process_with_vit(self, image):
        """Process image with ViT for classification"""
        if "vit_model" not in self.models or "vit_processor" not in self.models:
            return {"error": "ViT model not available"}
            
        # Prepare image for the model
        inputs = self.models["vit_processor"](images=image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = self.models["vit_model"](**inputs)
            logits = outputs.logits
        
        # Get top 5 predictions
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        top5_prob, top5_indices = torch.topk(probs, 5)
        
        results = []
        for i, (prob, idx) in enumerate(zip(top5_prob, top5_indices)):
            class_name = self.models["vit_model"].config.id2label[idx.item()]
            results.append({
                "rank": i+1,
                "class": class_name,
                "probability": round(prob.item(), 3)
            })
            
        return {"classifications": results}
    
    def _extract_product_info_from_vision(self, image, results):
        """Extract product information using LLM and vision results"""
        if self.llm is None:
            # Provide a basic extraction based on detection results only
            if "detections" in results and results["detections"]:
                detections = results["detections"]
                # Get the most confident detection
                top_detection = max(detections, key=lambda x: x["confidence"])
                return {
                    "product_type": top_detection["class"],
                    "confidence": top_detection["confidence"]
                }
            return {"product_type": "unknown"}
        
        # If we have an LLM, we can do more sophisticated extraction
        prompt = f"""Analyze this product image detection results and extract detailed product information.
        Detection results: {json.dumps(results)}
        
        Extract the following information in JSON format:
        - product_type: The category of the product (car, smartphone, laptop, etc.)
        - brand: The most likely brand of the product
        - model: Any model information that can be determined
        - color: The main color of the product
        - key_features: List of notable visual features
        
        Return only valid JSON format."""
        
        try:
            # Use LangChain's invoke method
            response = self.llm.invoke(prompt)
            
            # Extract content from LangChain response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Try to parse as JSON
            try:
                extracted = json.loads(response_text)
                return extracted
            except json.JSONDecodeError:
                # If LLM output is not valid JSON, extract key information using simple parsing
                lines = response_text.split('\n')
                extracted = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_').strip('"').strip("'")
                        # Clean up value more thoroughly
                        value = value.strip().rstrip(',').rstrip(';')
                        # Remove all types of quotes completely for cleaner values
                        value = value.strip('"').strip("'")
                        # Handle empty quoted strings
                        if value in ['""', "''", '""""', "''''", '']:
                            value = "Unknown"
                        # Clean up array-like strings - convert to proper arrays
                        if value.startswith('[') and value.endswith(']'):
                            try:
                                # Try to parse as actual array
                                import ast
                                parsed_array = ast.literal_eval(value)
                                if isinstance(parsed_array, list):
                                    value = parsed_array
                            except:
                                # If parsing fails, clean up the string
                                value = value.strip('[]').split(',') if value != '[]' else []
                        elif value.startswith('[') and not value.endswith(']'):
                            value = []
                        
                        if key and value not in ['null', 'None', None]:
                            extracted[key] = value
                return extracted
        except Exception as e:
            print(f"Error extracting product info: {e}")
            return {"product_type": "unknown", "error": str(e)}
    
    async def process(self, session_id, data):
        """Process product images to identify products and extract features"""
        self.log(session_id, "Starting image analysis to identify products...")
        
        results = {}
        product_info = {}
        
        # Process each image if available
        images = data.get("images", [])
        for i, img in enumerate(images):
            if img is None:
                continue
                
            image_key = f"image{i+1}"
            self.log(session_id, f"Processing {image_key}...")
            
            # Run object detection
            yolo_results = self._process_with_yolo(img)
            
            # Run classification
            vit_results = self._process_with_vit(img)
            
            # Combine results
            vision_results = {
                **yolo_results,
                **vit_results
            }
            
            # Extract product information from vision results
            info = self._extract_product_info_from_vision(img, vision_results)
            
            self.log(session_id, f"Identified product in {image_key}: {info.get('product_type', 'unknown')}")
            if "brand" in info:
                self.log(session_id, f"Detected brand: {info['brand']}")
                
            results[image_key] = {
                "vision_results": vision_results,
                "product_info": info
            }
            product_info[image_key] = info
            
        self.log(session_id, "Image processing completed")
        return {
            "vision_results": results,
            "product_info": product_info
        }


class FeatureExtractionAgent(BaseAgent):
    """Agent responsible for extracting detailed product features and specifications
    
    This agent analyzes image processing results and uses LLMs to extract detailed
    product specifications, features, and technical details.
    """
    
    def __init__(self, name="FeatureExtractionAgent"):
        super().__init__(name)
    
    def _extract_specifications(self, product_info):
        """Extract detailed specifications from product information"""
        if self.llm is None:
            # If no LLM is available, return basic specs from product info
            return product_info
            
        # Prepare prompt for specification extraction
        product_type = product_info.get("product_type", "unknown")
        prompt = f"""Based on this product information, generate a detailed list of specifications.
        Product information: {json.dumps(product_info)}
        
        For a {product_type}, extract or infer the following specifications.
        
        IMPORTANT: Return ONLY valid JSON format. Do not include any explanatory text before or after the JSON.
        
        Required JSON structure for a {product_type}:
        """
        
        # Add product-specific specification types based on product type
        if "car" in product_type.lower() or "vehicle" in product_type.lower():
            prompt += """{
            "make": "The manufacturer of the car",
            "model": "The model name", 
            "year": "Estimated year of manufacture",
            "body_type": "The body style (sedan, SUV, etc.)",
            "fuel_type": "The fuel type if identifiable",
            "engine": "Engine specifications if identifiable", 
            "color": "Exterior color",
            "features": ["List", "of", "visible", "features"]
        }
        
        Example: {"make": "Toyota", "model": "Camry", "year": "2020", "body_type": "Sedan", "fuel_type": "Gasoline", "engine": "2.5L", "color": "Blue", "features": ["LED headlights", "Alloy wheels"]}"""
        elif "phone" in product_type.lower() or "smartphone" in product_type.lower():
            prompt += """{
            "brand": "The manufacturer",
            "model": "The phone model",
            "screen_size": "Estimated screen size",
            "camera": "Visible camera specifications",
            "color": "Device color",
            "generation": "Device generation if identifiable",
            "features": ["List", "of", "visible", "features"]
        }
        
        Example: {"brand": "Apple", "model": "iPhone 14", "screen_size": "6.1 inches", "camera": "Dual camera", "color": "Blue", "generation": "14th", "features": ["Face ID", "Wireless charging"]}"""
        elif "laptop" in product_type.lower() or "computer" in product_type.lower():
            prompt += """{
            "brand": "The manufacturer",
            "model": "The computer model",
            "screen_size": "Estimated screen size",
            "form_factor": "Laptop, desktop, all-in-one, etc.",
            "color": "Device color",
            "features": ["List", "of", "visible", "features"]
        }
        
        Example: {"brand": "Dell", "model": "XPS 13", "screen_size": "13.3 inches", "form_factor": "Laptop", "color": "Silver", "features": ["Touchscreen", "Backlit keyboard"]}"""
        else:
            # Generic product specifications
            prompt += """{
            "brand": "The manufacturer if identifiable",
            "model": "The product model if identifiable", 
            "color": "Product color",
            "dimensions": "Estimated dimensions",
            "features": ["List", "of", "visible", "features"],
            "materials": "Visible materials used"
        }
        
        Example: {"brand": "Unknown", "model": "Unknown", "color": "Black", "dimensions": "Medium", "features": ["Modern design"], "materials": "Plastic"}"""
        
        try:
            # Use LangChain's invoke method
            response = self.llm.invoke(prompt)
            
            # Extract content from LangChain response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Try to parse as JSON
            try:
                specs = json.loads(response_text)
                return specs
            except json.JSONDecodeError:
                # If LLM output is not valid JSON, extract key information using simple parsing
                lines = response_text.split('\n')
                specs = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_').strip('"').strip("'")
                        # Clean up value more thoroughly
                        value = value.strip().rstrip(',').rstrip(';')
                        # Remove all types of quotes completely for cleaner values
                        value = value.strip('"').strip("'")
                        # Handle empty quoted strings
                        if value in ['""', "''", '""""', "''''", '']:
                            value = "Unknown"
                        # Clean up array-like strings - convert to proper arrays
                        if value.startswith('[') and value.endswith(']'):
                            try:
                                # Try to parse as actual array
                                import ast
                                parsed_array = ast.literal_eval(value)
                                if isinstance(parsed_array, list):
                                    value = parsed_array
                            except:
                                # If parsing fails, clean up the string
                                value = value.strip('[]').split(',') if value != '[]' else []
                        elif value.startswith('[') and not value.endswith(']'):
                            value = []
                        
                        if key and value not in ['null', 'None', None]:
                            specs[key] = value
                return specs
        except Exception as e:
            print(f"Error extracting specifications: {e}")
            return {"error": str(e)}
    
    def _get_feature_highlights(self, specs):
        """Extract key feature highlights from specifications"""
        if not specs or not isinstance(specs, dict):
            return []
            
        highlights = []
        
        # Extract key features based on product type
        product_type = specs.get("product_type", "").lower()
        
        if "car" in product_type or "vehicle" in product_type:
            # Highlight car features
            if "make" in specs and "model" in specs:
                highlights.append(f"{specs['make']} {specs['model']}")
            if "year" in specs:
                highlights.append(f"{specs['year']} model")
            if "engine" in specs:
                highlights.append(f"Engine: {specs['engine']}")
            if "body_type" in specs:
                highlights.append(f"{specs['body_type']} body style")
        
        elif "phone" in product_type or "smartphone" in product_type:
            # Highlight phone features
            if "brand" in specs and "model" in specs:
                highlights.append(f"{specs['brand']} {specs['model']}")
            if "screen_size" in specs:
                highlights.append(f"{specs['screen_size']} display")
            if "camera" in specs:
                highlights.append(f"Camera: {specs['camera']}")
        
        elif "laptop" in product_type or "computer" in product_type:
            # Highlight laptop features
            if "brand" in specs and "model" in specs:
                highlights.append(f"{specs['brand']} {specs['model']}")
            if "screen_size" in specs:
                highlights.append(f"{specs['screen_size']} display")
        
        # Generic highlights for any product
        if "features" in specs:
            if isinstance(specs["features"], list):
                highlights.extend(specs["features"][:3])  # Top 3 features
            elif isinstance(specs["features"], str):
                features = specs["features"].split(",")
                highlights.extend([f.strip() for f in features[:3]])  # Top 3 features
        
        # Add color as a feature if available
        if "color" in specs:
            highlights.append(f"{specs['color']} color")
            
        return highlights
    
    async def process(self, session_id, data):
        """Process product information to extract detailed specifications"""
        self.log(session_id, "Extracting detailed product specifications...")
        
        results = {}
        product_info = data.get("product_info", {})
        
        if not product_info:
            self.log(session_id, "No product information available for specification extraction")
            return {"specifications": {}}
            
        # Process each product
        for key, info in product_info.items():
            self.log(session_id, f"Extracting specifications for {key}...")
            
            # Extract detailed specifications
            specs = self._extract_specifications(info)
            
            # Get feature highlights
            highlights = self._get_feature_highlights(specs)
            
            # Log results
            if highlights:
                self.log(session_id, f"Key features for {key}: {', '.join(highlights[:3])}")
            
            results[key] = {
                "specifications": specs,
                "highlights": highlights
            }
            
        self.log(session_id, "Feature extraction completed")
        return {"specifications": results}


class ComparisonAgent(BaseAgent):
    """Agent responsible for comparing products based on their specifications
    
    This agent analyzes the specifications of multiple products and identifies
    the key differences, advantages, and disadvantages between them.
    """
    
    def __init__(self, name="ComparisonAgent"):
        super().__init__(name)
    
    def _compare_specifications(self, specs1, specs2):
        """Compare two sets of product specifications"""
        if not specs1 or not specs2:
            return {"error": "Insufficient specification data for comparison"}
            
        # If we have an LLM, use it for more sophisticated comparison
        if self.llm is not None:
            prompt = f"""Compare these two products based on their specifications.
            Product 1: {json.dumps(specs1)}
            Product 2: {json.dumps(specs2)}
            
            Provide a detailed comparison in JSON format with the following structure:
            {{"differences": [...], "product1_advantages": [...], "product2_advantages": [...], "summary": "..."}}
            
            - differences: List key differences between the products
            - product1_advantages: List advantages of Product 1 over Product 2
            - product2_advantages: List advantages of Product 2 over Product 1
            - summary: A concise summary of the comparison
            """
            
            try:
                # Use LangChain's invoke method
                response = self.llm.invoke(prompt)
                
                # Extract content from LangChain response
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # Try to parse as JSON
                try:
                    comparison = json.loads(response_text)
                    return comparison
                except json.JSONDecodeError:
                    # If LLM output is not valid JSON, extract key sections using simple parsing
                    lines = response_text.split('\n')
                    comparison = {}
                    
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip().lower().replace(' ', '_').strip('"').strip("'")
                            # Clean up value more thoroughly
                            value = value.strip().rstrip(',').rstrip(';')
                            # Remove all types of quotes completely for cleaner values
                            value = value.strip('"').strip("'")
                            # Handle empty quoted strings
                            if value in ['""', "''", '""""', "''''", '']:
                                value = "Unknown"
                            # Clean up array-like strings - convert to proper arrays
                            if value.startswith('[') and value.endswith(']'):
                                try:
                                    # Try to parse as actual array
                                    import ast
                                    parsed_array = ast.literal_eval(value)
                                    if isinstance(parsed_array, list):
                                        value = parsed_array
                                except:
                                    # If parsing fails, clean up the string
                                    value = value.strip('[]').split(',') if value != '[]' else []
                            elif value.startswith('[') and not value.endswith(']'):
                                value = []
                            
                            if key and value not in ['null', 'None', None]:
                                comparison[key] = value
                    
                    return comparison
            except Exception as e:
                print(f"Error in LLM comparison: {e}")
                # Fall back to simple comparison
                pass
        
        # Simple comparison logic as fallback
        differences = []
        product1_advantages = []
        product2_advantages = []
        
        # Identify common keys to compare
        all_keys = set(list(specs1.keys()) + list(specs2.keys()))
        
        # Exclude utility keys like 'error'
        exclude_keys = {'error', 'product_type', 'confidence'}
        compare_keys = all_keys - exclude_keys
        
        for key in compare_keys:
            val1 = specs1.get(key)
            val2 = specs2.get(key)
            
            if val1 is None and val2 is not None:
                differences.append(f"Product 2 has {key}: {val2}, but Product 1 doesn't")
                product2_advantages.append(f"Has {key}: {val2}")
            elif val1 is not None and val2 is None:
                differences.append(f"Product 1 has {key}: {val1}, but Product 2 doesn't")
                product1_advantages.append(f"Has {key}: {val1}")
            elif val1 != val2:
                differences.append(f"Different {key}: Product 1 has {val1}, Product 2 has {val2}")
                
                # Try to determine advantages based on common metrics
                if key in ['price', 'weight']:
                    # Lower is generally better
                    try:
                        num1 = float(str(val1).split()[0])
                        num2 = float(str(val2).split()[0])
                        if num1 < num2:
                            product1_advantages.append(f"Lower {key}: {val1}")
                        else:
                            product2_advantages.append(f"Lower {key}: {val2}")
                    except (ValueError, IndexError):
                        pass
                elif key in ['screen_size', 'storage', 'memory', 'ram', 'battery', 'capacity']:
                    # Higher is generally better
                    try:
                        num1 = float(str(val1).split()[0])
                        num2 = float(str(val2).split()[0])
                        if num1 > num2:
                            product1_advantages.append(f"Higher {key}: {val1}")
                        else:
                            product2_advantages.append(f"Higher {key}: {val2}")
                    except (ValueError, IndexError):
                        pass
        
        # Create a simple summary
        product1_type = specs1.get('product_type', 'Product 1')
        product2_type = specs2.get('product_type', 'Product 2')
        
        summary = f"Comparison between {product1_type} and {product2_type} reveals {len(differences)} key differences."
        
        if len(product1_advantages) > len(product2_advantages):
            summary += f" {product1_type} appears to have more advantages."
        elif len(product2_advantages) > len(product1_advantages):
            summary += f" {product2_type} appears to have more advantages."
        else:
            summary += " Both products have similar number of advantages."
            
        return {
            "differences": differences,
            "product1_advantages": product1_advantages,
            "product2_advantages": product2_advantages,
            "summary": summary
        }
    
    async def process(self, session_id, data):
        """Compare products based on their specifications"""
        self.log(session_id, "Starting product comparison analysis...")
        
        specifications = data.get("specifications", {})
        if len(specifications) < 2:
            self.log(session_id, "Not enough products to compare")
            return {"comparison": {"error": "Need at least two products to compare"}}
            
        # Get the product keys (image1, image2, etc.)
        product_keys = list(specifications.keys())
        
        if len(product_keys) > 2:
            self.log(session_id, f"Found {len(product_keys)} products, comparing the first two only")
            product_keys = product_keys[:2]
            
        # Get the specifications for each product
        product1_specs = specifications.get(product_keys[0], {}).get("specifications", {})
        product2_specs = specifications.get(product_keys[1], {}).get("specifications", {})
        
        # Perform comparison
        comparison = self._compare_specifications(product1_specs, product2_specs)
        
        # Log comparison results
        if "differences" in comparison:
            num_diff = len(comparison["differences"])
            self.log(session_id, f"Found {num_diff} key differences between the products")
            
            # Log a few example differences
            if num_diff > 0:
                for i, diff in enumerate(comparison["differences"][:3]):
                    self.log(session_id, f"Difference {i+1}: {diff}")
                    
        # Log summary if available
        if "summary" in comparison:
            self.log(session_id, f"Comparison summary: {comparison['summary']}")
            
        self.log(session_id, "Comparison analysis completed")
        return {
            "comparison": comparison,
            "product_keys": product_keys
        }


class RecommendationAgent(BaseAgent):
    """Agent responsible for providing purchase recommendations
    
    This agent analyzes product comparisons and provides personalized recommendations
    based on the user's needs and preferences.
    """
    
    def __init__(self, name="RecommendationAgent"):
        super().__init__(name)
    
    def _generate_recommendation(self, comparison, product_keys, specifications):
        """Generate a purchase recommendation based on comparison results"""
        if not comparison or "error" in comparison:
            return {
                "recommendation": "Insufficient data to make a recommendation",
                "reason": "Could not compare the products",
                "confidence": 0.0
            }
            
        # Get product information
        product1_key = product_keys[0]
        product2_key = product_keys[1]
        product1_specs = specifications.get(product1_key, {}).get("specifications", {})
        product2_specs = specifications.get(product2_key, {}).get("specifications", {})
        
        product1_type = product1_specs.get("product_type", product1_key)
        product2_type = product2_specs.get("product_type", product2_key)
        
        # If we have an LLM, use it for more sophisticated recommendation
        if self.llm is not None:
            prompt = f"""Based on this product comparison, provide a purchase recommendation.
            Product 1: {json.dumps(product1_specs)}
            Product 2: {json.dumps(product2_specs)}
            Comparison: {json.dumps(comparison)}
            
            Provide a recommendation in JSON format with the following structure:
            {{"recommended_product": "1 or 2", "recommendation": "...", "reason": "...", "confidence": 0.0-1.0, "use_cases": [...]}}
            
            - recommended_product: Either "1" or "2" indicating which product is recommended
            - recommendation: A concise recommendation statement
            - reason: The main reason for the recommendation
            - confidence: A confidence score between 0 and 1
            - use_cases: List of ideal use cases for the recommended product
            """
            
            try:
                # Use LangChain's invoke method
                response = self.llm.invoke(prompt)
                
                # Extract content from LangChain response
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # Try to parse as JSON
                try:
                    recommendation = json.loads(response_text)
                    return recommendation
                except json.JSONDecodeError:
                    # If LLM output is not valid JSON, extract key information using simple parsing
                    lines = response_text.split('\n')
                    recommendation = {}
                    
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip().lower().replace(' ', '_').strip('"').strip("'")
                            # Clean up value more thoroughly
                            value = value.strip().rstrip(',').rstrip(';')
                            # Remove all types of quotes completely for cleaner values
                            value = value.strip('"').strip("'")
                            # Handle empty quoted strings
                            if value in ['""', "''", '""""', "''''", '']:
                                value = "Unknown"
                            # Clean up array-like strings - convert to proper arrays
                            if value.startswith('[') and value.endswith(']'):
                                try:
                                    # Try to parse as actual array
                                    import ast
                                    parsed_array = ast.literal_eval(value)
                                    if isinstance(parsed_array, list):
                                        value = parsed_array
                                except:
                                    # If parsing fails, clean up the string
                                    value = value.strip('[]').split(',') if value != '[]' else []
                            elif value.startswith('[') and not value.endswith(']'):
                                value = []
                            
                            if key and value not in ['null', 'None', None]:
                                recommendation[key] = value
                    
                    return recommendation
            except Exception as e:
                print(f"Error in LLM recommendation: {e}")
                # Fall back to simple recommendation
                pass
        
        # Simple recommendation logic as fallback
        product1_advantages = comparison.get("product1_advantages", [])
        product2_advantages = comparison.get("product2_advantages", [])
        
        # Count advantages
        p1_count = len(product1_advantages)
        p2_count = len(product2_advantages)
        
        # Simple confidence calculation
        total_advantages = p1_count + p2_count
        if total_advantages == 0:
            confidence = 0.5  # Can't determine
        else:
            # Maximum confidence is 0.95, minimum is 0.55
            max_confidence = 0.95
            min_confidence = 0.55
            base_confidence = max(p1_count, p2_count) / total_advantages if total_advantages > 0 else 0.5
            confidence = min_confidence + base_confidence * (max_confidence - min_confidence)
            
        # Make recommendation
        if p1_count > p2_count:
            recommended = "1"
            recommendation = f"{product1_type} is recommended over {product2_type}"
            reason = f"It has {p1_count} advantages compared to {p2_count} for the alternative"
        elif p2_count > p1_count:
            recommended = "2"
            recommendation = f"{product2_type} is recommended over {product1_type}"
            reason = f"It has {p2_count} advantages compared to {p1_count} for the alternative"
        else:
            # Equal advantages, slight preference for first product
            recommended = "1"
            recommendation = f"Both {product1_type} and {product2_type} appear equally matched"
            reason = "Consider your specific needs as both have similar advantages"
            confidence = 0.5  # Equal confidence
            
        # Determine use cases based on advantages
        use_cases = []
        
        if recommended == "1" and product1_advantages:
            # Extract use cases from product 1's advantages
            use_cases = [f"When {adv.lower()}" for adv in product1_advantages[:3]]
        elif recommended == "2" and product2_advantages:
            # Extract use cases from product 2's advantages
            use_cases = [f"When {adv.lower()}" for adv in product2_advantages[:3]]
        
        # Add generic use case if none found
        if not use_cases:
            use_cases = [f"General {product1_type if recommended == '1' else product2_type} usage"]
            
        return {
            "recommended_product": recommended,
            "recommendation": recommendation,
            "reason": reason,
            "confidence": round(confidence, 2),
            "use_cases": use_cases
        }
    
    async def process(self, session_id, data):
        """Generate purchase recommendations based on product comparison"""
        self.log(session_id, "Generating product recommendations...")
        
        # Get comparison data
        comparison = data.get("comparison", {})
        product_keys = data.get("product_keys", [])
        specifications = data.get("specifications", {})
        
        if not comparison or not product_keys or len(product_keys) < 2:
            self.log(session_id, "Insufficient data for recommendation")
            return {"recommendation": {"error": "Not enough data to generate recommendation"}}
            
        # Generate recommendation
        recommendation = self._generate_recommendation(comparison, product_keys, specifications)
        
        # Log recommendation
        self.log(session_id, f"Recommendation: {recommendation.get('recommendation', 'No recommendation')}")
        self.log(session_id, f"Reason: {recommendation.get('reason', 'No reason provided')}")
        
        # Log confidence
        confidence = recommendation.get('confidence', 0)
        try:
            # Handle various confidence formats
            if isinstance(confidence, str):
                # Extract first valid number from string like '0.70.70.70...'
                confidence_clean = confidence.split('.')[0] + '.' + confidence.split('.')[1] if '.' in confidence else confidence
                confidence = float(confidence_clean)
            confidence = float(confidence)
            confidence_percent = f"{int(confidence * 100)}%"
        except (ValueError, IndexError):
            confidence_percent = "Unknown"
        self.log(session_id, f"Confidence in recommendation: {confidence_percent}")
        
        self.log(session_id, "Recommendation generation completed")
        return {"recommendation": recommendation}


class ProductComparisonCoordinator:
    """Main coordinator for the product comparison multi-agent system
    
    This class orchestrates the entire product comparison workflow by managing
    all the specialized agents and their interactions.
    """
    
    def __init__(self):
        # Initialize all agents
        self.image_processor = ImageProcessingAgent()
        self.feature_extractor = FeatureExtractionAgent()
        self.comparison_agent = ComparisonAgent()
        self.recommendation_agent = RecommendationAgent()
        
    async def process_images(self, session_id, images, session_metadata=None):
        """Process images through the entire multi-agent workflow
        
        Args:
            session_id: Unique session identifier
            images: List of image data (PIL Images or numpy arrays)
            session_metadata: Optional dictionary with additional session information
            
        Returns:
            Dictionary containing the final analysis results
        """
        # Initialize session if it doesn't exist
        if session_manager.get_status(session_id) is None:
            session_manager.create_session(session_id)
        
        # Set default metadata if not provided
        if session_metadata is None:
            session_metadata = {}
        
        # Get analysis type from metadata
        analysis_type = session_metadata.get('analysis_type', 'info')
        
        session_manager.set_status(session_id, "processing")
        session_manager.add_message(session_id, f"Starting product {analysis_type} analysis")
        
        try:
            # Step 1: Process images with Image Processing Agent
            session_manager.add_message(session_id, "Step 1: Analyzing product images...")
            image_results = await self.image_processor.process(session_id, {"images": images})
            
            # Check if we have enough products to compare
            product_info = image_results.get("product_info", {})
            if len(product_info) < 1:
                session_manager.add_message(session_id, "Error: No products detected in images")
                session_manager.set_status(session_id, "error")
                return {"error": "No products detected in images"}
                
            # Step 2: Extract features with Feature Extraction Agent
            session_manager.add_message(session_id, "Step 2: Extracting product specifications...")
            try:
                feature_results = await self.feature_extractor.process(session_id, image_results)
            except Exception as e:
                session_manager.add_message(session_id, f"Warning: Feature extraction failed: {str(e)}")
                feature_results = {"specifications": {"error": f"Feature extraction failed: {str(e)}"}}
            
            # Step 3: Compare products with Comparison Agent if we have multiple products
            comparison_results = {}
            if len(product_info) >= 2:
                session_manager.add_message(session_id, "Step 3: Comparing products...")
                try:
                    comparison_results = await self.comparison_agent.process(
                        session_id, 
                        {**feature_results, "specifications": feature_results.get("specifications", {})}
                    )
                except Exception as e:
                    session_manager.add_message(session_id, f"Warning: Comparison failed: {str(e)}")
                    comparison_results = {"comparison": {"error": f"Comparison failed: {str(e)}"}}
                
                # Step 4: Generate recommendation with Recommendation Agent
                session_manager.add_message(session_id, "Step 4: Generating purchase recommendation...")
                try:
                    recommendation_results = await self.recommendation_agent.process(
                        session_id,
                        {**comparison_results, "specifications": feature_results.get("specifications", {})}
                    )
                except Exception as e:
                    session_manager.add_message(session_id, f"Warning: Recommendation failed: {str(e)}")
                    recommendation_results = {"recommendation": {"error": f"Recommendation failed: {str(e)}"}}
            else:
                session_manager.add_message(session_id, "Skipping comparison: Only one product detected")
                comparison_results = {"comparison": {"error": "Need at least two products to compare"}}
                recommendation_results = {"recommendation": {"error": "Need at least two products to compare"}}
                
            # Tailor results based on analysis type
            final_results = {
                "status": "completed"
            }
            
            # Include results based on analysis type
            if analysis_type == 'info':
                final_results["productInfo"] = image_results.get("product_info", {})
                final_results["specifications"] = feature_results.get("specifications", {})
                
            elif analysis_type == 'compare':
                final_results["comparison"] = comparison_results.get("comparison", {})
                final_results["productInfo"] = image_results.get("product_info", {})
                
            elif analysis_type == 'value':
                # Value analysis combines specs and comparison data
                final_results["valueAnalysis"] = {
                    "priceA": "$" + str(1000 + int(hash(str(session_id)) % 500)),  # Mock price for demo
                    "valueScoreA": 7 + (int(hash(str(session_id)) % 3)),  # Mock score between 7-9
                    "analysis": "Based on the specifications and market positioning, this product offers good value for money."
                }
                if len(product_info) >= 2:
                    final_results["valueAnalysis"]["priceB"] = "$" + str(1200 + int(hash(str(session_id + "B")) % 500))
                    final_results["valueAnalysis"]["valueScoreB"] = 6 + (int(hash(str(session_id + "B")) % 4))
                
            elif analysis_type == 'recommend':
                if recommendation_results.get("recommendation", {}).get("error"):
                    # Create a mock recommendation if only one product
                    if len(product_info) == 1:
                        product_name = next(iter(product_info.values())).get("name", "Product")
                        final_results["recommendation"] = {
                            "recommendedProduct": product_name,
                            "reason": "This is the only product analyzed and appears to meet standard quality benchmarks.",
                            "confidence": 0.85,
                            "alternatives": [
                                {"name": "Similar model with higher storage", "reason": "If you need more storage capacity"},
                                {"name": "Budget alternative", "reason": "If price is your primary concern"}
                            ],
                            "buyingTips": [
                                "Wait for seasonal sales for the best price",
                                "Check for warranty terms before purchasing"
                            ]
                        }
                    else:
                        final_results["recommendation"] = recommendation_results.get("recommendation", {})
                else:
                    final_results["recommendation"] = recommendation_results.get("recommendation", {})
            
            # Always include basic product info and vision results for context
            final_results["vision_results"] = image_results.get("vision_results", {})
            if "productInfo" not in final_results:  # Don't duplicate if already added
                final_results["productInfo"] = image_results.get("product_info", {})
            
            # Set final results in session manager
            session_manager.set_final_result(session_id, final_results)
            session_manager.set_status(session_id, "completed")
            session_manager.add_message(session_id, "Product comparison analysis completed successfully")
            
            return final_results
            
        except Exception as e:
            error_msg = f"Error during product comparison: {str(e)}"
            session_manager.add_message(session_id, error_msg)
            session_manager.set_status(session_id, "error")
            return {"error": error_msg}
    
    def get_session_messages(self, session_id):
        """Get all messages for a session"""
        return session_manager.get_messages(session_id)
    
    def get_session_result(self, session_id):
        """Get the final result for a session"""
        return session_manager.get_final_result(session_id)
    
    def get_session_status(self, session_id):
        """Get the status of a session"""
        return session_manager.get_status(session_id)


# Helper function to create a coordinator instance
def get_product_comparison_coordinator():
    """Get a singleton instance of the ProductComparisonCoordinator"""
    if not hasattr(get_product_comparison_coordinator, "_instance"):
        get_product_comparison_coordinator._instance = ProductComparisonCoordinator()
    return get_product_comparison_coordinator._instance


# Helper function to convert base64 image data to PIL Image
def decode_base64_image(base64_data):
    """Convert base64 image data to PIL Image"""
    try:
        # Check if the base64 data includes a data URL prefix
        if base64_data.startswith('data:image'):
            # Extract the actual base64 data after the comma
            base64_data = base64_data.split(',', 1)[1]
            
        # Decode base64 data
        image_data = base64.b64decode(base64_data)
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None
