import requests
import cv2
import numpy as np
import io

def create_test_image():
    # Create a dummy image (green leaf-like)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[:] = [34, 139, 34] # Forest Green
    # Add a "diseased" spot (brown)
    cv2.circle(img, (256, 256), 50, (42, 42, 165), -1) 
    
    # Encode
    _, buf = cv2.imencode('.jpg', img)
    return io.BytesIO(buf.tobytes())

def test_api():
    url = "http://localhost:8000/analyze"
    print(f"Testing API at {url}...")
    
    try:
        # Create dummy image
        img_bytes = create_test_image()
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            print("✓ Success! Response received.")
            data = response.json()
            print("\nResponse Data:")
            print(f"Disease Severity: {data['disease']['severity']}")
            print(f"Diseased Area: {data['disease']['diseased_area_percent']}%")
            print(f"Pest Count: {data['pest']['count']}")
            print(f"Pest Severity: {data['pest']['severity']}")
            
            if data['disease']['mask_b64']:
                print("✓ Disease mask present in response")
            else:
                print("x No disease mask returned")
                
        else:
            print(f"x Failed with status {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"x Connection Error: {e}")
        print("Ensure the backend is running (uvicorn main:app --reload)")

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) > 1:
        # Use provided image path
        img_path = sys.argv[1]
        if os.path.exists(img_path):
            print(f"Testing with image: {img_path}")
            # Read bytes
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            
            url = "http://localhost:8000/analyze"
            try:
                files = {'file': (os.path.basename(img_path), img_bytes, 'image/jpeg')}
                response = requests.post(url, files=files)
                print(f"Status Code: {response.status_code}")
                print(response.json())
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"File not found: {img_path}")
    else:
        test_api()
