import os
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = 'plant_disease_model.h5'
CLASSES_PATH = 'class_names.json'
UPLOAD_FOLDER = 'uploads'
DB_PATH = os.environ.get('DB_PATH', 'agriguard.db')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('temp_uploads', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# AI Model (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────
model = None
class_names = {}

def load_ai_assets():
    global model, class_names
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        print(f"Loading AI model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        with open(CLASSES_PATH, 'r') as f:
            class_names = {int(k): v for k, v in json.load(f).items()}
        print("✅ AI Model loaded successfully!")
    else:
        print("⚠️  AI Model not found. /predict will return an error.")

# ─────────────────────────────────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS fields (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            name                TEXT    NOT NULL,
            location            TEXT,
            status              TEXT    DEFAULT 'unknown',
            latest_moisture     REAL,
            moisture_updated_at TEXT,
            created_at          TEXT    DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT    NOT NULL UNIQUE,
            password   TEXT    NOT NULL,
            role       TEXT    NOT NULL DEFAULT 'farmer',
            created_at TEXT    DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS scans (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            field_id       INTEGER NOT NULL,
            image_path     TEXT    NOT NULL,
            result_disease TEXT,
            confidence     REAL,
            recommendation TEXT,
            created_at     TEXT    DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            action     TEXT    NOT NULL,
            created_at TEXT    DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    # Seed the single eggplant field
    conn.execute(
        "INSERT OR IGNORE INTO fields (id, name, location, status) VALUES (1, 'Eggplant Field', 'Main Zone', 'unknown')"
    )
    # Seed the admin account (plain password; login supports plain & hashed)
    conn.execute(
        "INSERT OR IGNORE INTO users (id, username, password, role) VALUES (1, 'admin', 'admin123', 'owner')"
    )
    conn.commit()
    conn.close()
    print("✅ Database initialised.")

def rows_to_list(rows):
    return [dict(row) for row in rows]

def row_to_dict(row):
    return dict(row) if row else None

# ─────────────────────────────────────────────────────────────────────────────
# AI helper – used by both /predict and /upload_image.php
# ─────────────────────────────────────────────────────────────────────────────
def generate_recommendation(disease_name):
    d = disease_name.lower()
    if 'healthy'    in d: return "Plant is healthy. Maintain regular drip irrigation, monitor soil moisture, and continue routine nutrient management."
    if 'insect'     in d or 'pest' in d: return "Insect pest damage detected. Prune infested shoots. Spray neem oil (2-3%) or Bacillus thuringiensis early in the morning."
    if 'spot'       in d: return "Fungal leaf spot detected. Remove infected lower leaves. Avoid overhead watering and apply copper-based or Mancozeb fungicide."
    if 'mosaic'     in d: return "Mosaic virus detected. Remove infected plants immediately. Control aphids/whiteflies with yellow sticky traps or insecticidal soap."
    if 'small leaf' in d or 'little leaf' in d: return "Little Leaf Disease (phytoplasma). Discard stunted plants. Apply systemic insecticides (Dimethoate) to control leafhopper vectors."
    if 'white mold' in d or 'mold' in d: return "White mold detected. Prune affected stems, reduce moisture, and apply Trichoderma bio-fungicide around the base."
    if 'wilt'       in d: return "Wilt disease detected. Remove wilted plants with root soil. Ensure proper drainage and treat with copper oxychloride."
    return "Crop anomaly detected. Isolate affected area, sanitise tools, and consult a local agricultural extension specialist."

def run_ai_prediction(filepath):
    """Run the AI model on an image file. Returns a result dict."""
    if model is None:
        return {"error": "AI Model not loaded."}

    img = Image.open(filepath).convert('RGB')
    img_resized = img.resize((224, 224))

    # Color heuristic – reject non-plant images early
    hsv = np.array(img_resized.convert('HSV'))
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    plant_pixels = (H >= 10) & (H <= 120) & (S >= 25) & (V >= 25)
    if np.mean(plant_pixels) < 0.05:
        return {
            "status": "unknown",
            "disease": "Not a Plant / Unrecognized",
            "recommendation": "The camera did not detect enough plant colours. Ensure a crop leaf is clearly in frame.",
            "confidence": 0.0
        }

    # Image enhancements
    from PIL import ImageEnhance
    enhanced = ImageEnhance.Contrast(img_resized).enhance(1.2)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.5)

    # Test-Time Augmentation (TTA)
    variations = [enhanced, enhanced.rotate(90), enhanced.transpose(Image.FLIP_LEFT_RIGHT), enhanced.rotate(180)]
    batch = np.array([np.array(v) / 255.0 for v in variations])
    preds = model.predict(batch)
    avg = np.mean(preds, axis=0)

    # Healthy-bias penalty
    for idx, name in class_names.items():
        if 'healthy' in name.lower():
            avg[idx] *= 0.80
    avg = avg / np.sum(avg)

    idx = int(np.argmax(avg))
    confidence = float(avg[idx])

    if confidence < 0.75:
        return {
            "status": "unknown",
            "disease": "Not a Plant / Unrecognized",
            "recommendation": "Could not confidently identify a crop. Point the camera clearly at a plant leaf and try again.",
            "confidence": confidence
        }

    disease = class_names.get(idx, "Unknown")
    is_healthy = 'healthy' in disease.lower()
    return {
        "status": "healthy" if is_healthy else "attention_needed",
        "disease": disease,
        "recommendation": generate_recommendation(disease),
        "confidence": confidence
    }

# ─────────────────────────────────────────────────────────────────────────────
# Routes – Authentication
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/login.php')
def login():
    username = request.args.get('username', '').strip()
    password = request.args.get('password', '').strip()
    if not username or not password:
        return jsonify({"status": "error", "message": "Username and password required"})

    db   = get_db()
    user = db.execute("SELECT id, username, password, role FROM users WHERE username = ?", (username,)).fetchone()
    if not user:
        return jsonify({"status": "error", "message": "User not found"})

    stored = user['password']
    valid  = (password == stored)          # plain-text fallback (seed account)
    if not valid:
        try:
            valid = check_password_hash(stored, password)
        except Exception:
            valid = False

    if valid:
        try:
            db.execute("INSERT INTO audit_logs (user_id, action) VALUES (?, 'login')", (user['id'],))
            db.commit()
        except Exception:
            pass
        return jsonify({"status": "success", "user": {"id": user['id'], "username": user['username'], "role": user['role']}})

    return jsonify({"status": "error", "message": "Invalid password"})

# ─────────────────────────────────────────────────────────────────────────────
# Routes – Fields
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/get_fields.php')
def get_fields():
    db   = get_db()
    rows = db.execute("SELECT * FROM fields ORDER BY name ASC").fetchall()
    return jsonify({"status": "success", "data": rows_to_list(rows)})

@app.route('/get_field_status.php')
def get_field_status():
    field_id = request.args.get('field_id', type=int)
    if not field_id:
        return jsonify({"status": "error", "message": "field_id is required"})

    db = get_db()
    # Auto-cleanup scans older than 2 months
    db.execute("DELETE FROM scans WHERE field_id = ? AND created_at < datetime('now', '-2 months')", (field_id,))
    db.commit()

    field = row_to_dict(db.execute("SELECT * FROM fields WHERE id = ?", (field_id,)).fetchone())
    if not field:
        return jsonify({"status": "error", "message": "Field not found"})

    scans = rows_to_list(db.execute("SELECT * FROM scans WHERE field_id = ? ORDER BY created_at DESC", (field_id,)).fetchall())

    # Derive status from most recent valid scan
    field['status'] = 'unknown'
    if scans:
        latest = scans[0]
        if latest['result_disease'] == 'Not a Plant / Unrecognized':
            for scan in scans:
                if scan['result_disease'] != 'Not a Plant / Unrecognized':
                    field['status'] = 'healthy' if 'healthy' in scan['result_disease'].lower() else 'attention_needed'
                    break
        else:
            field['status'] = 'healthy' if 'healthy' in latest['result_disease'].lower() else 'attention_needed'

    field['recent_scans'] = scans
    field['total_scans']  = len(scans)
    return jsonify({"status": "success", "data": field})

# ─────────────────────────────────────────────────────────────────────────────
# Routes – Moisture
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/save_moisture.php', methods=['GET', 'POST'])
def save_moisture():
    field_id = request.values.get('field_id', 1, type=int)
    moisture = request.values.get('moisture', type=float)
    if moisture is None:
        return jsonify({"status": "error", "message": "moisture value is required"})

    now = datetime.utcnow().isoformat()
    db  = get_db()
    db.execute("UPDATE fields SET latest_moisture = ?, moisture_updated_at = ? WHERE id = ?", (moisture, now, field_id))
    db.commit()
    return jsonify({"status": "success", "moisture": moisture})

# ─────────────────────────────────────────────────────────────────────────────
# Routes – Farmers
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/manage_farmers.php')
def manage_farmers():
    action = request.args.get('action', '')
    db     = get_db()

    if action == 'list':
        rows = db.execute("SELECT id, username, created_at FROM users WHERE role = 'farmer'").fetchall()
        return jsonify({"status": "success", "farmers": rows_to_list(rows)})

    if action == 'add':
        username = request.args.get('username', '').strip()
        password = request.args.get('password', '').strip()
        if not username or not password:
            return jsonify({"status": "error", "message": "Username and password required"})
        try:
            db.execute("INSERT INTO users (username, password, role) VALUES (?, ?, 'farmer')",
                       (username, generate_password_hash(password)))
            db.commit()
            return jsonify({"status": "success", "message": "Farmer account created"})
        except Exception:
            return jsonify({"status": "error", "message": "Username might already exist"})

    if action == 'delete':
        user_id = request.args.get('id', type=int)
        if not user_id:
            return jsonify({"status": "error", "message": "ID required"})
        db.execute("DELETE FROM users WHERE id = ? AND role = 'farmer'", (user_id,))
        db.commit()
        return jsonify({"status": "success", "message": "Farmer account deleted"})

    return jsonify({"status": "error", "message": "Invalid action"})

# ─────────────────────────────────────────────────────────────────────────────
# Routes – Audit Logs
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/audit_log.php')
def audit_log():
    user_id = request.args.get('user_id', type=int)
    action  = request.args.get('action', '').strip()
    if not user_id or not action:
        return jsonify({"status": "error", "message": "Missing parameters"})

    db = get_db()
    db.execute("INSERT INTO audit_logs (user_id, action) VALUES (?, ?)", (user_id, action))
    db.commit()
    return jsonify({"status": "success"})

@app.route('/get_audit_logs.php')
def get_audit_logs():
    user_id = request.args.get('user_id', type=int)
    if not user_id:
        return jsonify({"status": "error", "message": "user_id is required"})

    db   = get_db()
    user = db.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user or user['role'] != 'owner':
        return jsonify({"status": "error", "message": "Unauthorized"})

    rows = db.execute("""
        SELECT a.id, a.action, a.created_at, u.username
        FROM audit_logs a
        JOIN users u ON a.user_id = u.id
        ORDER BY a.created_at DESC
        LIMIT 100
    """).fetchall()
    return jsonify({"status": "success", "data": rows_to_list(rows)})

# ─────────────────────────────────────────────────────────────────────────────
# Routes – Image Upload + AI Scan
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/upload_image.php', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image in request"}), 400

    file     = request.files['image']
    field_id = request.form.get('field_id', 1, type=int)

    filename = secure_filename(file.filename or 'capture.jpg')
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        result = run_ai_prediction(filepath)
    except Exception as e:
        os.remove(filepath)
        return jsonify({"status": "error", "message": str(e)}), 500

    disease        = result.get('disease', 'Unknown')
    recommendation = result.get('recommendation', '')
    confidence     = result.get('confidence', 0.0)
    field_status   = result.get('status', 'unknown')

    if 'error' not in result:
        confidence_pct = round(confidence * 100, 2) if 0 < confidence <= 1.0 else round(confidence, 2)
    else:
        confidence_pct = 0

    db = get_db()
    db.execute(
        "INSERT INTO scans (field_id, image_path, result_disease, confidence, recommendation) VALUES (?, ?, ?, ?, ?)",
        (field_id, filepath, disease, confidence_pct, recommendation)
    )
    if field_status != 'unknown':
        db.execute("UPDATE fields SET status = ? WHERE id = ?", (field_status, field_id))
    db.commit()

    return jsonify({
        "status": "success",
        "message": "Rover image scanned successfully",
        "data": {
            "disease":      disease,
            "confidence":   confidence_pct,
            "recommendation": recommendation,
            "field_status": field_status,
            "image_url":    filepath
        }
    })

# ─────────────────────────────────────────────────────────────────────────────
# Routes – AI Predict (direct POST, kept for compatibility)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "AI Model not loaded."}), 500
    if 'image' not in request.files:
        return jsonify({"error": "No image in request"}), 400

    file     = request.files['image']
    filename = secure_filename(file.filename or 'upload.jpg')
    filepath = os.path.join('temp_uploads', filename)
    file.save(filepath)

    try:
        result = run_ai_prediction(filepath)
    except Exception as e:
        result = {"error": str(e)}
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify(result)

# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    load_ai_assets()
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Agri-Guard server running on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
