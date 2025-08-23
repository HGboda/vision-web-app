---
title: Vision Llm Agent
emoji: 🌖
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
license: gpl-3.0
---

# Vision LLM Agent - Object Detection with AI Assistant

A multi-model object detection and image classification demo with LLM-based AI assistant for answering questions about detected objects. This project uses YOLOv8, DETR, and ViT models for vision tasks, and TinyLlama for natural language processing. The application includes a secure login system to protect access to the AI features.

## 🎬 Demo

![Vision LLM Agent Demo](demo.gif)

*Live demo showing product comparison analysis with image upload, real-time processing, and detailed results across multiple analysis tabs.*

## ✨ Features

- **Multi-Model Object Detection**: YOLOv8, DETR, and ViT models for comprehensive image analysis
- **Product Comparison**: AI-powered comparison of multiple products with detailed analysis
- **Real-time Processing**: Live streaming of analysis progress with SSE (Server-Sent Events)
- **Secure Authentication**: Protected access with Flask-Login session management
- **Modern UI**: React frontend with Material-UI components
- **Vector Database**: Similarity search and object recognition capabilities
- **OpenAI Integration**: GPT-5-mini for intelligent analysis and recommendations

## Project Architecture

This project follows a phased development approach:

### Phase 0: PoC with Gradio (Original)
- Simple Gradio interface with multiple object detection models
- Uses Hugging Face's free tier for model hosting
- Easy to deploy to Hugging Face Spaces

### Phase 1: Service Separation (Implemented)
- Backend: Flask API with model inference endpoints
- REST API endpoints for model inference
- JSON responses with detection results and performance metrics

### Phase 2: UI Upgrade (Implemented)
- Modern React frontend with Material-UI components
- Improved user experience with responsive design
- Separate frontend and backend architecture

### Phase 3: CI/CD & Testing (Planned)
- GitHub Actions for automated testing and deployment
- Comprehensive test suite with pytest and ESLint
- Automatic rebuilds on Hugging Face Spaces

## How to Run

### Option 1: Original Gradio App
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Gradio app:
   ```bash
   python app.py
   ```

3. Open your browser and go to the URL shown in the terminal (typically `http://127.0.0.1:7860`)

### Option 2: React Frontend with Flask Backend
1. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the Flask backend server:
   ```bash
   python api.py
   ```

3. In a separate terminal, navigate to the frontend directory:
   ```bash
   cd frontend
   ```

4. Install frontend dependencies:
   ```bash
   npm install
   ```

5. Start the React development server:
   ```bash
   npm start
   ```

6. Open your browser and go to `http://localhost:3000`

## Models Used

- **YOLOv8**: Fast and accurate object detection
- **DETR**: DEtection TRansformer for object detection
- **ViT**: Vision Transformer for image classification
- **TinyLlama**: For natural language processing and question answering about detected objects

## Authentication

The application includes a secure login system to protect access to all features:

- **Default Credentials**:
  - Username: `admin` / Password: `admin123`
  - Username: `user` / Password: `user123`

- **Login Process**:
  - All routes and API endpoints are protected with Flask-Login
  - Users must authenticate before accessing any features
  - Session management handles login state persistence

- **Security Features**:
  - Password protection for all API endpoints and UI pages
  - Session-based authentication with secure cookies
  - Configurable secret key via environment variables

## API Endpoints

The Flask backend provides the following API endpoints (all require authentication):

- `GET /api/status` - Check the status of the API and available models
- `POST /api/detect/yolo` - Detect objects using YOLOv8
- `POST /api/detect/detr` - Detect objects using DETR
- `POST /api/classify/vit` - Classify images using ViT
- `POST /api/analyze` - Analyze images with LLM assistant
- `POST /api/similar-images` - Find similar images in the vector database
- `POST /api/add-to-collection` - Add images to the vector database
- `POST /api/add-detected-objects` - Add detected objects to the vector database
- `POST /api/search-similar-objects` - Search for similar objects in the vector database

All POST endpoints accept form data with an 'image' field containing the image file.

## Deployment

### Gradio App
The Gradio app is designed to be easily deployed to Hugging Face Spaces:

1. Create a new Space on Hugging Face
2. Select Gradio as the SDK
3. Push this repository to the Space's git repository
4. The app will automatically deploy

### React + Flask App
For the React + Flask version, you'll need to:

1. Build the React frontend:
   ```bash
   cd frontend
   npm run build
   ```

2. Serve the static files from a web server or cloud hosting service
3. Deploy the Flask backend to a server that supports Python
