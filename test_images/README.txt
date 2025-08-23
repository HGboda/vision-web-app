# Product Comparison API Testing Instructions

To test the product comparison API, place test images in this directory. The test images can be:

1. Cars (e.g., car1.jpg, car2.jpg)
2. Electronics like smartphones or laptops (e.g., phone1.jpg, phone2.jpg)
3. Any other products you want to compare

## Testing Steps

1. Start the Flask app: `python app.py`
2. In a separate terminal, run the test script:
   `python test_product_comparison_api.py --images test_images/image1.jpg test_images/image2.jpg`

## Expected Results

The test script will:
- Send images to the API endpoint
- Connect to the SSE stream to receive real-time updates
- Display the final analysis results

If successful, you should see structured output with product details, comparisons, and recommendations.
