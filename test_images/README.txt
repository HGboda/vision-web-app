# Product Comparison API Testing Instructions

This directory contains test images for the vision LLM agent.

Note: Test images should be added locally for testing purposes.
The repository does not include binary image files due to Hugging Face Space limitations.

For testing, add your own images:
- car1.jpg: Test vehicle image 1
- car2.jpg: Test vehicle image 2

These images can be used to test:
1. Object detection and classification
2. Image analysis and description
3. Product comparison features
4. Multi-image processing capabilities

Images should be in JPEG format and optimized for web usage while maintaining sufficient quality for computer vision tasks.

## Testing Steps

1. Add test images to this directory locally
2. Start the Flask app: `python api.py`
3. Access the web interface at http://localhost:7860
4. Use the Product Comparison tab to upload and compare images

## Expected Results

The application will provide detailed analysis and comparison of uploaded images.
- Send images to the API endpoint
- Connect to the SSE stream to receive real-time updates
- Display the final analysis results

If successful, you should see structured output with product details, comparisons, and recommendations.
