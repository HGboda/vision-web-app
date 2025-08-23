#!/usr/bin/env python3
"""
Product Comparison API Test Script

This script tests the product comparison API endpoints with local image files.
It supports both the /api/product/compare/start and /api/product/compare/stream endpoints.

Requirements:
- pip install requests pillow sseclient-py

Usage:
    python test_product_comparison_api.py --images path/to/image1.jpg path/to/image2.jpg

Example:
    python test_product_comparison_api.py --images test_images/car1.jpg test_images/car2.jpg
"""

import requests
import base64
import json
import time
import os
import sys
from io import BytesIO
from PIL import Image
import argparse

try:
    import sseclient
except ImportError:
    print("Missing required package: sseclient-py")
    print("Please install with: pip install sseclient-py")
    sys.exit(1)

def load_and_encode_image(image_path):
    """Load an image file and convert to base64 string"""
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format=img.format or "JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return img_str
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def test_product_comparison_api(base_url, image_paths, timeout=300):
    """Test the product comparison API with provided images"""
    print(f"Testing product comparison API with {len(image_paths)} images")
    
    # 1. Encode images as base64
    images = []
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Error: Image file {path} not found")
            return
        
        encoded = load_and_encode_image(path)
        if encoded:
            images.append(encoded)
            print(f"Successfully encoded image: {path}")
    
    if not images:
        print("No valid images to process")
        return
    
    # 2. Start a comparison session
    start_url = f"{base_url}/api/product/compare/start"
    print(f"Sending request to {start_url}")
    
    try:
        response = requests.post(
            start_url,
            json={"images": images}
        )
        
        if not response.ok:
            print(f"Error starting comparison: HTTP {response.status_code}")
            print(response.text)
            return
        
        data = response.json()
        session_id = data.get('session_id')
        
        if not session_id:
            print("Error: No session ID received")
            return
        
        print(f"Session started with ID: {session_id}")
        print(f"Status: {data.get('status')}")
        
        # 3. Connect to the streaming endpoint
        stream_url = f"{base_url}/api/product/compare/stream/{session_id}"
        print(f"Connecting to stream at {stream_url}")
        
        headers = {'Accept': 'text/event-stream'}
        response = requests.get(stream_url, headers=headers, stream=True)
        
        client = sseclient.SSEClient(response)
        
        start_time = time.time()
        result = None
        
        print("\n--- Streaming Updates ---")
        try:
            for event in client.events():
                current_time = time.time()
                if current_time - start_time > timeout:
                    print("Timeout reached. Ending stream.")
                    break
                
                try:
                    data = json.loads(event.data)
                    
                    if 'message' in data:
                        print(f"Message: {data['message']}")
                    
                    if 'status' in data:
                        print(f"Status update: {data['status']}")
                    
                    if 'result' in data:
                        print("\n--- Final Results ---")
                        result = data['result']
                        print(json.dumps(result, indent=2))
                        break
                    
                    if 'error' in data:
                        print(f"Error: {data['error']}")
                        break
                        
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from event data: {event.data}")
                    continue
        except KeyboardInterrupt:
            print("Stream monitoring interrupted by user")
        
        return result
        
    except Exception as e:
        print(f"Error in API test: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Product Comparison API')
    parser.add_argument('--url', default='http://localhost:5000', help='Base URL for the API')
    parser.add_argument('--images', nargs='+', required=True, help='Paths to images for testing')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds')
    
    args = parser.parse_args()
    
    test_product_comparison_api(args.url, args.images, args.timeout)
