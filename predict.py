"""
AI vs Real Image Classifier - Prediction Module
Uses pre-trained Hugging Face model for detection.
Model: jacoballessio/ai-image-detect-distilled (~50MB, 11.8M parameters)
"""

import numpy as np
from PIL import Image
import os

# Hugging Face transformers
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import torch

# Configuration
MODEL_NAME = "jacoballessio/ai-image-detect-distilled"
CACHE_DIR = "./model_cache"

# Global model variables
_classifier = None
_processor = None
_model = None


def load_model():
    """Load the pre-trained Hugging Face model"""
    global _classifier, _processor, _model
    
    if _classifier is not None:
        return _classifier
    
    print(f"Loading model: {MODEL_NAME}...")
    print("This may take a moment on first run (downloading ~50MB)...")
    
    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    try:
        # Load using pipeline (simpler approach)
        _classifier = pipeline(
            "image-classification",
            model=MODEL_NAME,
            cache_dir=CACHE_DIR,
            device=-1  # CPU (-1) or GPU (0)
        )
        print("Model loaded successfully!")
        return _classifier
        
    except Exception as e:
        print(f"Pipeline loading failed: {e}")
        print("Trying alternative loading method...")
        
        # Alternative: Load model and processor separately
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        _model = AutoModelForImageClassification.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        _model.eval()
        
        print("Model loaded successfully (alternative method)!")
        return None


def preprocess_image(image_path):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    return img


def predict_image(image_path, use_tta=False):
    """
    Predict if an image is AI-generated or real
    
    Args:
        image_path: Path to the image file
        use_tta: Whether to use test-time augmentation (horizontal flip)
    
    Returns:
        Dictionary containing prediction results
    """
    global _classifier, _processor, _model
    
    # Load model if not already loaded
    classifier = load_model()
    
    # Load image
    img = preprocess_image(image_path)
    
    if classifier is not None:
        # Using pipeline
        if use_tta:
            # Original prediction
            result1 = classifier(img)
            
            # Flipped prediction
            img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            result2 = classifier(img_flipped)
            
            # Average the scores
            scores = {}
            for r in result1:
                scores[r['label']] = r['score']
            for r in result2:
                scores[r['label']] = (scores.get(r['label'], 0) + r['score']) / 2
            
            # Find best prediction
            results = [{'label': k, 'score': v} for k, v in scores.items()]
            results.sort(key=lambda x: x['score'], reverse=True)
        else:
            results = classifier(img)
    else:
        # Using manual approach
        inputs = _processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = _model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get labels
        labels = _model.config.id2label
        results = []
        for idx, prob in enumerate(probs[0]):
            results.append({
                'label': labels[idx],
                'score': float(prob)
            })
        results.sort(key=lambda x: x['score'], reverse=True)
    
    # Parse results
    # Model outputs: 'artificial' or 'human' (or similar labels)
    ai_score = 0.0
    real_score = 0.0
    
    for r in results:
        label_lower = r['label'].lower()
        if 'artificial' in label_lower or 'ai' in label_lower or 'fake' in label_lower or 'generated' in label_lower:
            ai_score = r['score']
        elif 'human' in label_lower or 'real' in label_lower or 'authentic' in label_lower:
            real_score = r['score']
    
    # Determine prediction
    if ai_score > real_score:
        prediction_label = "AI Generated"
        confidence = ai_score * 100
    else:
        prediction_label = "Real"
        confidence = real_score * 100
    
    result = {
        'prediction': prediction_label,
        'confidence': float(confidence),
        'ai_probability': float(ai_score),
        'real_probability': float(real_score),
        'raw_score': float(ai_score),
        'used_tta': use_tta,
        'model_used': MODEL_NAME
    }
    
    return result


def predict_batch(image_paths):
    """
    Predict multiple images at once
    
    Args:
        image_paths: List of image file paths
    
    Returns:
        List of prediction dictionaries
    """
    results = []
    
    for image_path in image_paths:
        try:
            result = predict_image(image_path, use_tta=False)
            results.append(result)
        except Exception as e:
            results.append({
                'prediction': 'Error',
                'confidence': 0.0,
                'ai_probability': 0.0,
                'real_probability': 0.0,
                'error': str(e)
            })
    
    return results


def get_model_info():
    """Get model information"""
    load_model()
    
    info = {
        'model_type': 'Hugging Face Pre-trained (ViT Distilled)',
        'model_name': MODEL_NAME,
        'input_size': '224x224',
        'total_parameters': '11.8M (distilled)',
        'model_size': '~50MB',
        'class_names': ['AI Generated', 'Real']
    }
    
    return info


def main():
    """Test the prediction module"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [--tta]")
        print(f"\nModel: {MODEL_NAME}")
        print("This model detects AI-generated images vs real images.")
        sys.exit(1)
    
    image_path = sys.argv[1]
    use_tta = '--tta' in sys.argv
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"\nAnalyzing image: {image_path}")
    print(f"Using TTA: {use_tta}")
    print(f"Model: {MODEL_NAME}\n")
    
    try:
        result = predict_image(image_path, use_tta=use_tta)
        
        print("=" * 50)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"AI Probability: {result['ai_probability']*100:.2f}%")
        print(f"Real Probability: {result['real_probability']*100:.2f}%")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()