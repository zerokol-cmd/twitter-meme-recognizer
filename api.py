"""
Flask API Server for Image Analysis
Provides REST endpoints for image classification using ResNet152
Saves classified images to disk for future training.
"""

import io
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import base64
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
CAPTURED_DATA_DIR = "captured_images"

# Global model cache
_model_cache = None
_class_names_cache = None
_device_cache = None


def save_image_to_disk(image, class_name, prefix="img"):
    """
    Saves the PIL image to captured_images/<class_name>/<timestamp>_<prefix>.jpg
    """
    try:
        # 1. Create directory: captured_images/twitter/
        directory = os.path.join(CAPTURED_DATA_DIR, class_name)
        os.makedirs(directory, exist_ok=True)

        # 2. Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Clean prefix to be filename safe
        clean_prefix = str(prefix).replace("/", "_").replace("\\", "_")
        filename = f"{timestamp}_{clean_prefix}.jpg"
        
        file_path = os.path.join(directory, filename)

        # 3. Save
        # Convert to RGB just in case (though we do this in endpoints usually)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image.save(file_path, "JPEG", quality=95)
        print(f"[Saver] Saved to: {file_path}")
        return file_path
    except Exception as e:
        print(f"[Saver] Error saving image: {e}")
        return None


def load_model(weights_path="twitter_classifier.pth", device=None):
    """Load the ResNet152 model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = models.ResNet152_Weights.IMAGENET1K_V2
    model = models.resnet152(weights=weights)
    
    # determine number of classes from data/
    try:
        dataset = ImageFolder(root="data/")
        num_classes = len(dataset.classes)
        class_names = dataset.classes
    except Exception:
        # fallback if data folder not present
        num_classes = 2
        # Assuming typical use case based on previous conversation
        class_names = ["other", "twitter"] 

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # wrap model (must match training setup)
    class ResNet152WithDropout(nn.Module):
        def __init__(self, base_model, num_classes):
            super().__init__()
            self.base = base_model
            
        def forward(self, x):
            return self.base(x)
    
    wrapped_model = ResNet152WithDropout(model, num_classes)
    wrapped_model.to(device)

    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        try:
            wrapped_model.load_state_dict(state)
        except Exception:
            # allow loading partial/state-dict variations
            wrapped_model.load_state_dict(state, strict=False)
    else:
        print(f"Warning: weights file '{weights_path}' not found. Using randomly initialized model.")

    wrapped_model.eval()
    return wrapped_model, class_names, device


# Image preprocessing pipeline
_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_model_cached(weights_path="twitter_classifier.pth"):
    """Get cached model to avoid reloading"""
    global _model_cache, _class_names_cache, _device_cache
    
    if _model_cache is None:
        _model_cache, _class_names_cache, _device_cache = load_model(weights_path)
    
    return _model_cache, _class_names_cache, _device_cache


# ============== API Endpoints ==============

@app.route('/api/status', methods=['GET'])
def api_status():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'Twitter Image Classifier',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    Analyze a single image
    """
    try:
        image = None
        
        # Try to get image from file upload
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(file.stream).convert('RGB')
        # Try to get image from JSON base64
        elif request.is_json:
            data = request.get_json()
            if 'image' in data:
                image_data = base64.b64decode(data['image'].split(',')[-1])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        if image is None:
            return jsonify({
                'error': 'No image provided',
                'success': False
            }), 400
        
        # Get model
        model, class_names, device = get_model_cached()
        
        # Prepare input
        input_tensor = _PREPROCESS(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Create response
        probabilities = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        top_class = max(probabilities.items(), key=lambda x: x[1])
        predicted_class = top_class[0]
        
        # --- SAVE IMAGE ---
        save_image_to_disk(image, predicted_class, prefix="web_upload")
        # ------------------

        result = {
            'success': True,
            'class': predicted_class,
            'confidence': top_class[1],
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat(),
            'model': 'resnet152'
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f'API Error: {str(e)}')
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/telegram/analyze', methods=['POST'])
def api_telegram_analyze():
    """
    Analyze image from Telegram extension
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'error': 'Missing image data',
                'success': False
            }), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'].split(',')[-1])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            return jsonify({
                'error': f'Invalid image format: {str(e)}',
                'success': False
			}), 400
		
        # Get model
        model, class_names, device = get_model_cached()
        
        # Prepare input
        input_tensor = _PREPROCESS(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Create response
        probabilities = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
        top_class = max(probabilities.items(), key=lambda x: x[1])
        predicted_class = top_class[0]

        # --- SAVE IMAGE ---
        # Use Message ID if available in data, else 'tg_unknown'
        msg_id = data.get('id', 'unknown_id')
        save_image_to_disk(image, predicted_class, prefix=f"tg_{msg_id}")
        # ------------------
        
        result = {
            'success': True,
            'class': predicted_class,
            'confidence': top_class[1],
            'probabilities': probabilities,
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'model': 'resnet152'
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f'API Error: {str(e)}')
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/api/analyze-batch', methods=['POST'])
def api_analyze_batch():
    """
    Analyze multiple images
    """
    try:
        data = request.get_json()
        
        if not data or 'images' not in data:
            return jsonify({
                'error': 'Missing images array',
                'success': False
            }), 400
        
        images = data['images']
        results = []
        model, class_names, device = get_model_cached()
        
        for img_item in images:
            try:
                # Decode
                if isinstance(img_item, str):
                    image_data = base64.b64decode(img_item.split(',')[-1])
                    item_id = "batch"
                elif isinstance(img_item, dict):
                    image_data = base64.b64decode(img_item['image'].split(',')[-1])
                    item_id = img_item.get('id', 'batch')
                else:
                    continue
                
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Predict
                input_tensor = _PREPROCESS(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
                probabilities = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
                top_class = max(probabilities.items(), key=lambda x: x[1])
                predicted_class = top_class[0]

                # --- SAVE IMAGE ---
                save_image_to_disk(image, predicted_class, prefix=f"batch_{item_id}")
                # ------------------
                
                result = {
                    'id': item_id,
                    'class': predicted_class,
                    'confidence': top_class[1],
                    'probabilities': probabilities
                }
                results.append(result)
            except Exception as e:
                results.append({
                    'id': img_item.get('id') if isinstance(img_item, dict) else None,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


if __name__ == '__main__':
    print('Starting Flask API server...')
    print(f'Images will be saved to: ./{CAPTURED_DATA_DIR}/')
    print('Available endpoints:')
    print('  GET  /api/status')
    print('  POST /api/analyze - Send single image')
    print('  POST /api/telegram/analyze - From Telegram extension')
    print('  POST /api/analyze-batch - Send multiple images')
    print('\nServer running on http://0.0.0.0:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)