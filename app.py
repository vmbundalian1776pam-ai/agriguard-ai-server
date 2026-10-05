import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Configuration
MODEL_PATH = 'plant_disease_model.h5'
CLASSES_PATH = 'class_names.json'
UPLOAD_FOLDER = 'temp_uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables for model and classes
model = None
class_names = {}

def load_ai_assets():
    global model, class_names
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        
        with open(CLASSES_PATH, 'r') as f:
            class_names_dict = json.load(f)
            # JSON keys are strings, but our model outputs integer indices. Convert keys to int.
            class_names = {int(k): v for k, v in class_names_dict.items()}
        print("✅ AI Model and Classes loaded successfully!")
    else:
        print("⚠️ WARNING: AI Model not found. You must run train.py first!")

def generate_recommendation(disease_name):
    disease_lower = disease_name.lower()
    
    if 'healthy' in disease_lower:
        return "Plant is healthy and showing vigorous foliage. Maintain regular drip irrigation, monitor soil moisture, and continue routine weed and nutrient management."
    elif 'insect' in disease_lower or 'pest' in disease_lower:
        return "Insect pest damage detected (e.g., shoot borers, aphids, or flea beetles). Prune and destroy infested shoots. Spray organic neem oil (2-3%) or use recommended biological insecticides (e.g., Bacillus thuringiensis) early in the morning."
    elif 'spot' in disease_lower:
        return "Fungal leaf spot detected (Cercospora/Alternaria). Remove infected lower leaves to prevent spore splash. Avoid overhead watering to keep foliage dry, and apply a copper-based or Mancozeb fungicide."
    elif 'mosaic' in disease_lower:
        return "Mosaic virus detected. Viruses cannot be cured once inside the plant. Remove and safely dispose of infected plants immediately. Control sap-sucking insect vectors (aphids/whiteflies) using yellow sticky traps or insecticidal soap."
    elif 'small leaf' in disease_lower or 'little leaf' in disease_lower:
        return "Little Leaf Disease detected (caused by phytoplasma). Uproot and discard severely stunted plants. Control leafhopper insect vectors by applying systemic insecticides (like Dimethoate) or neem-based sprays."
    elif 'white mold' in disease_lower or 'mold' in disease_lower:
        return "White mold fungus detected (Sclerotinia). Immediately prune affected stems and clean up fallen plant debris. Reduce soil moisture, increase sunlight exposure, and apply a bio-fungicide (Trichoderma) around the base."
    elif 'wilt' in disease_lower:
        return "Wilt disease detected (bacterial or fungal vascular blockage). Remove wilted plants along with root soil to prevent spread. Ensure proper field drainage, avoid over-irrigation, and treat root zones with copper oxychloride."
    else:
        return "Crop anomaly detected. Isolate affected leaves, sanitize farm tools, ensure proper soil drainage, and consult a local agricultural extension specialist."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "AI Model is not loaded. Train the model first using train.py."}), 500
        
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Preprocess the image exactly how it was trained
            img = Image.open(filepath).convert('RGB')
            img_resized = img.resize((224, 224))
            
            # --- COLOR HEURISTIC FILTER (The "Is this actually a plant?" check) ---
            # Neural networks are easily tricked by out-of-context images (like a white cat)
            # because they try to force every image into one of the 7 leaf categories.
            # To fix this, we analyze the actual colors in the image using HSV.
            hsv_img = np.array(img_resized.convert('HSV'))
            H = hsv_img[:, :, 0] # Hue (Color: 0-255)
            S = hsv_img[:, :, 1] # Saturation (Grey vs Colorful: 0-255)
            V = hsv_img[:, :, 2] # Value (Dark vs Bright: 0-255)
            
            # In PIL's 0-255 scale: ~10 to 120 covers Browns, Yellows, and Greens
            # S > 25 excludes pure white/grey (like the cat or a wall)
            # V > 25 excludes pitch black shadows
            plant_pixels = (H >= 10) & (H <= 120) & (S >= 25) & (V >= 25)
            plant_ratio = np.mean(plant_pixels)
            
            # If less than 5% of the image contains plant-like colors, reject it immediately
            if plant_ratio < 0.05:
                os.remove(filepath)
                return jsonify({
                    "status": "unknown",
                    "disease": "Not a Plant / Unrecognized",
                    "recommendation": "AI Rejected: The camera did not detect enough plant colors (green, yellow, or brown). Please ensure a crop leaf is clearly in the frame.",
                    "confidence": 0.0
                })

            # --- IMAGE ENHANCEMENT ---
            # Boost contrast and sharpness to highlight small details like insect bites
            from PIL import ImageEnhance
            img_enhanced = ImageEnhance.Contrast(img_resized).enhance(1.2)
            img_enhanced = ImageEnhance.Sharpness(img_enhanced).enhance(1.5)

            # --- TEST-TIME AUGMENTATION (TTA) ---
            # Generate 4 variations of the image. This forces the AI to look at the leaf from multiple angles
            img_1 = img_enhanced
            img_2 = img_enhanced.rotate(90)
            img_3 = img_enhanced.transpose(Image.FLIP_LEFT_RIGHT)
            img_4 = img_enhanced.rotate(180)
            
            variations = [img_1, img_2, img_3, img_4]
            batch_array = []
            for v in variations:
                arr = np.array(v) / 255.0
                batch_array.append(arr)
            
            batch_array = np.array(batch_array)
            
            # Predict all 4 at once
            predictions = model.predict(batch_array)
            
            # Average the probabilities across all 4 angles
            avg_predictions = np.mean(predictions, axis=0)
            
            # --- HEALTHY BIAS PENALTY ---
            # AI models often default to "healthy" if they are lazy. 
            # We manually penalize the 'healthy' score by 20% to force the model to only guess healthy if it's ABSOLUTELY certain.
            for idx, name in class_names.items():
                if 'healthy' in name.lower():
                    avg_predictions[idx] *= 0.80 
            
            # Re-normalize so percentages add up to 100%
            avg_predictions = avg_predictions / np.sum(avg_predictions)
            
            predicted_class_idx = np.argmax(avg_predictions)
            confidence = float(avg_predictions[predicted_class_idx])
            
            # --- OUT OF DISTRIBUTION FILTER ---
            # CNNs will try to classify anything (even a wall or shoe) into one of the 7 leaf categories.
            # Usually, when it's a random object, the confidence drops because it doesn't match perfectly.
            # We enforce a strict 75% confidence threshold.
            if confidence < 0.75:
                disease_name = "Not a Plant / Unrecognized"
                status = "unknown"
                recommendation = "The AI could not confidently identify a crop in this image. Please ensure the rover camera is clearly pointed at a plant leaf and try scanning again."
            else:
                disease_name = class_names.get(predicted_class_idx, "Unknown")
                is_healthy = 'healthy' in disease_name.lower()
                status = "healthy" if is_healthy else "attention_needed"
                recommendation = generate_recommendation(disease_name)
            
            # Cleanup temp file
            os.remove(filepath)
            
            return jsonify({
                "status": status,
                "disease": disease_name,
                "recommendation": recommendation,
                "confidence": confidence
            })
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_ai_assets()
    # Run on port 5000 so it doesn't conflict with XAMPP (Port 80)
    print("🚀 Starting Local AI Server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
