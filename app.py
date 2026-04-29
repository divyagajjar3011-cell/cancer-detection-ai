from flask import Flask, render_template, request, jsonify
import os
import uuid
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ==================== CONFIGURATION ====================
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==================== LOAD MODEL ====================
try:
    model = load_model("model.h5", compile=False)
    print("✅ Model loaded successfully!")
    print("✅ Model ready for predictions!")
except Exception as e:
    print(f"⚠️ Model load error: {e}")
    print("⚠️ Please make sure 'model.h5' file is in the same directory")
    model = None

# ==================== HELPER FUNCTIONS ====================
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_cancer(image_path):
    """
    Predict cancer from image
    Adjust target_size according to your model's training size
    """
    try:
        # Load image - CHANGE target_size (96, 224, 150) according to your model
        img = image.load_img(image_path, target_size=(96, 96))
        
        # Convert to array and preprocess
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        
        # Make prediction
        prediction = model.predict(img_array)
        raw_score = float(prediction[0][0])
        
        # Binary classification
        # If raw_score > 0.5 -> Cancer, else -> No Cancer
        if raw_score > 0.5:
            confidence = round(raw_score * 100, 2)
            return {
                "success": True,
                "prediction": "Malignant - Cancer Detected",
                "confidence": confidence,
                "class": "cancer",
                "raw_score": raw_score
            }
        else:
            confidence = round((1 - raw_score) * 100, 2)
            return {
                "success": True,
                "prediction": "Benign - No Cancer Detected",
                "confidence": confidence,
                "class": "no-cancer",
                "raw_score": 1 - raw_score
            }
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# ==================== ROUTES ====================
@app.route("/")
def home():
    """Home page"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Handle image upload and return prediction
    """
    # Check if model is loaded
    if model is None:
        return jsonify({
            "success": False, 
            "error": "Model not loaded. Please check model.h5 file."
        }), 500
    
    # Check if file is present in request
    if "file" not in request.files:
        return jsonify({
            "success": False, 
            "error": "No file uploaded"
        }), 400
    
    file = request.files["file"]
    
    # Check if file is selected
    if file.filename == "":
        return jsonify({
            "success": False, 
            "error": "No file selected"
        }), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({
            "success": False, 
            "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    try:
        # Save file with unique name
        original_filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + original_filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_cancer(filepath)
        
        # Optional: Delete file after prediction to save space
        # os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False, 
            "error": f"Prediction error: {str(e)}"
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

# ==================== RUN APP ====================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Starting MedScan Pro Server")
    print("="*50)
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🤖 Model loaded: {model is not None}")
    print(f"🌐 Server running at: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5000)

    # target_size aapke model ke according set karo
img = image.load_img(image_path, target_size=(96, 96))  # 96, 224, ya 150