# Vision Web App - Object Detection Demo

A multi-model object detection and image classification demo using YOLOv8, DETR, and ViT models. This project is designed to showcase different computer vision models for a hiring demonstration.

## Project Architecture

This project follows a phased development approach:

### Phase 0: PoC with Gradio (Current)
- Simple Gradio interface with multiple object detection models
- Uses Hugging Face's free tier for model hosting
- Easy to deploy to Hugging Face Spaces

### Phase 1: Service Separation (Planned)
- Backend: FastAPI with ONNX runtime optimization
- REST API endpoints for model inference
- Docker containerization

### Phase 2: UI Upgrade (Planned)
- Modern React frontend with TypeScript and Tailwind CSS
- Improved user experience and portfolio-quality UI
- Static frontend build served by backend

### Phase 3: CI/CD & Testing (Planned)
- GitHub Actions for automated testing and deployment
- Comprehensive test suite with pytest and ESLint
- Automatic rebuilds on Hugging Face Spaces

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Gradio app:
   ```bash
   python app.py
   ```

3. Open your browser and go to the URL shown in the terminal (typically `http://127.0.0.1:7860`)

## Models Used

- **YOLOv8**: Fast and accurate object detection
- **DETR**: DEtection TRansformer for object detection
- **ViT**: Vision Transformer for image classification

## Deployment to Hugging Face Spaces

This app is designed to be easily deployed to Hugging Face Spaces for free hosting:

1. Create a new Space on Hugging Face
2. Select Gradio as the SDK
3. Push this repository to the Space's git repository
4. The app will automatically deploy
