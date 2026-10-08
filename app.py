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

SUPABASE_URL = "https://bvczwcpjjcwymgkwywgb.supabase.co"
SUPABASE_KEY = "sb_publishable_ETOMeJANE6gfsixt98nuRg_WYVZL3pQ"

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH    = 'plant_disease_model.h5'
CLASSES_PATH  = 'class_names.json'
UPLOAD_FOLDER = 'uploads'
TEMP_FOLDER   = 'temp_uploads'
DB_PATH       = os.environ.get('DB_PATH', 'agriguard.db')
DATABASE_URL  = os.environ.get('DATABASE_URL')   # Set this on Render for Supabase

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER,   exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Database driver selection
# ─────────────────────────────────────────────────────────────────────────────
if DATABASE_URL:
    import psycopg2
    import psycopg2.extras
    print("🐘 Using PostgreSQL (Supabase)")
    PLACEHOLDER = '%s'
else:
    print("🗄️  Using SQLite (local)")
    PLACEHOLDER = '?'


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        if DATABASE_URL:
            db = psycopg2.connect(
                DATABASE_URL,
                cursor_factory=psycopg2.extras.RealDictCursor,
                sslmode='require'
            )
        else:
            db = sqlite3.connect(DB_PATH)
            db.row_factory = sqlite3.Row
        g._database = db
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def query(sql, params=None, fetchone=False, fetchall=False, commit=False):
    """Run a SQL query, abstracting the SQLite/Postgres difference."""
    db  = get_db()
    sql = sql.replace('?', PLACEHOLDER)       # unify placeholders
    if DATABASE_URL:
        cur = db.cursor()
        cur.execute(sql, params or ())
        if commit:
            db.commit()
        if fetchone:
            row = cur.fetchone()
            return dict(row) if row else None
        if fetchall:
            return [dict(r) for r in cur.fetchall()]
        return cur
    else:
        cur = db.execute(sql, params or ())
        if commit:
            db.commit()
        if fetchone:
            row = cur.fetchone()
            return dict(row) if row else None
        if fetchall:
            return [dict(r) for r in cur.fetchall()]
        return cur


def init_db():
    try:
        query("ALTER TABLE fields ADD COLUMN latest_temperature REAL;", commit=True)
    except Exception:
        pass
    """Create all tables and seed initial data."""
    if DATABASE_URL:
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        cur  = conn.cursor()
        statements = [
            """CREATE TABLE IF NOT EXISTS fields (
                id                  SERIAL PRIMARY KEY,
                name                TEXT    NOT NULL,
                location            TEXT,
                status              TEXT    DEFAULT 'unknown',
                latest_moisture     REAL,
                latest_temperature  REAL,
                moisture_updated_at TEXT,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS users (
                id         SERIAL PRIMARY KEY,
                username   TEXT    NOT NULL UNIQUE,
                password   TEXT    NOT NULL,
                role       TEXT    NOT NULL DEFAULT 'farmer',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS scans (
                id             SERIAL PRIMARY KEY,
                field_id       INTEGER NOT NULL,
                image_path     TEXT    NOT NULL,
                result_disease TEXT,
                confidence     REAL,
                recommendation TEXT,
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS audit_logs (
                id         SERIAL PRIMARY KEY,
                user_id    INTEGER NOT NULL,
                action     TEXT    NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            # Seed data (ON CONFLICT = do nothing if already exists)
            """INSERT INTO fields (id, name, location, status)
               VALUES (1, 'Eggplant Field', 'Main Zone', 'unknown')
               ON CONFLICT DO NOTHING""",
            """INSERT INTO users (id, username, password, role)
               VALUES (1, 'admin', 'admin123', 'owner')
               ON CONFLICT DO NOTHING""",
        ]
        for stmt in statements:
            cur.execute(stmt)
        conn.commit()
        cur.close()
        conn.close()
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS fields (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                name                TEXT    NOT NULL,
                location            TEXT,
                status              TEXT    DEFAULT "unknown",
                latest_moisture     REAL,
                latest_temperature  REAL,
                moisture_updated_at TEXT,
                created_at          TEXT    DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT    NOT NULL UNIQUE,
                password   TEXT    NOT NULL,
                role       TEXT    NOT NULL DEFAULT "farmer",
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
        conn.execute("INSERT OR IGNORE INTO fields (id, name, location, status) VALUES (1, 'Eggplant Field', 'Main Zone', 'unknown')")
        conn.execute("INSERT OR IGNORE INTO users (id, username, password, role) VALUES (1, 'admin', 'admin123', 'owner')")
        conn.commit()
        conn.close()

    print("✅ Database initialised.")

# ─────────────────────────────────────────────────────────────────────────────
# AI Model
# ─────────────────────────────────────────────────────────────────────────────
model       = None
class_names = {}

def load_ai_assets():
    global model, class_names
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        print(f"Loading AI model...")
        model = load_model(MODEL_PATH)
        with open(CLASSES_PATH, 'r') as f:
            class_names = {int(k): v for k, v in json.load(f).items()}
        print("✅ AI Model loaded!")
    else:
        print("⚠️  AI Model not found.")

def generate_recommendation(disease_name):
    d = disease_name.lower()
    if 'healthy'    in d: return "Plant is healthy. Maintain regular irrigation and routine nutrient management."
    if 'insect'     in d or 'pest' in d: return "Insect pest damage detected. Prune infested shoots and spray neem oil (2–3%)."
    if 'spot'       in d: return "Fungal leaf spot detected. Remove infected leaves and apply copper-based fungicide."
    if 'mosaic'     in d: return "Mosaic virus detected. Remove infected plants and control aphids/whiteflies."
    if 'small leaf' in d or 'little leaf' in d: return "Little Leaf Disease detected. Discard stunted plants and control leafhopper vectors."
    if 'white mold' in d or 'mold' in d: return "White mold detected. Prune affected stems and apply Trichoderma bio-fungicide."
    if 'wilt'       in d: return "Wilt disease detected. Remove wilted plants and treat with copper oxychloride."
    return "Crop anomaly detected. Isolate affected area and consult an agricultural extension specialist."

def run_ai_prediction(filepath):
    if model is None:
        return {"error": "AI Model not loaded."}
    img        = Image.open(filepath).convert('RGB')
    img_resized = img.resize((224, 224))

    hsv = np.array(img_resized.convert('HSV'))
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    if np.mean((H >= 10) & (H <= 120) & (S >= 20) & (V >= 20)) < 0.01:
        return {"status": "unknown", "disease": "Not a Plant / Unrecognized",
                "recommendation": "The camera did not detect enough plant colours. Ensure a crop leaf is clearly in frame.",
                "confidence": 0.0}

    from PIL import ImageEnhance
    enhanced = ImageEnhance.Contrast(img_resized).enhance(1.2)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.5)

    variations = [enhanced, enhanced.rotate(90), enhanced.transpose(Image.FLIP_LEFT_RIGHT), enhanced.rotate(180)]
    batch = np.array([np.array(v) / 255.0 for v in variations])
    avg   = np.mean(model.predict(batch), axis=0)

    for idx, name in class_names.items():
        if 'healthy' in name.lower():
            avg[idx] *= 0.80
    avg = avg / np.sum(avg)

    idx        = int(np.argmax(avg))
    confidence = float(avg[idx])

    if confidence < 0.75:
        return {"status": "unknown", "disease": "Not a Plant / Unrecognized",
                "recommendation": "Could not confidently identify a crop. Point the camera at a plant leaf and retry.",
                "confidence": confidence}

    disease = class_names.get(idx, "Unknown")
    return {"status": "healthy" if 'healthy' in disease.lower() else "attention_needed",
            "disease": disease,
            "recommendation": generate_recommendation(disease),
            "confidence": confidence}

# ─────────────────────────────────────────────────────────────────────────────
# Routes — Auth
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/login.php')
def login():
    username = request.args.get('username', '').strip()
    password = request.args.get('password', '').strip()
    if not username or not password:
        return jsonify({"status": "error", "message": "Username and password required"})

    user = query("SELECT id, username, password, role FROM users WHERE username = ?",
                 (username,), fetchone=True)
    if not user:
        return jsonify({"status": "error", "message": "User not found"})

    stored = user['password']
    valid  = (password == stored)
    if not valid:
        try:
            valid = check_password_hash(stored, password)
        except Exception:
            valid = False

    if valid:
        try:
            query("INSERT INTO audit_logs (user_id, action) VALUES (?, 'login')",
                  (user['id'],), commit=True)
        except Exception:
            pass
        return jsonify({"status": "success",
                        "user": {"id": user['id'], "username": user['username'], "role": user['role']}})
    return jsonify({"status": "error", "message": "Invalid password"})

# ─────────────────────────────────────────────────────────────────────────────
# Routes — Fields
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/get_fields.php')
def get_fields():
    rows = query("SELECT * FROM fields ORDER BY name ASC", fetchall=True)
    return jsonify({"status": "success", "data": rows})

@app.route('/get_field_status.php')
def get_field_status():
    field_id = request.args.get('field_id', type=int)
    if not field_id:
        return jsonify({"status": "error", "message": "field_id is required"})

    if DATABASE_URL:
        query("DELETE FROM scans WHERE field_id = ? AND created_at < NOW() - INTERVAL '2 months'",
              (field_id,), commit=True)
    else:
        query("DELETE FROM scans WHERE field_id = ? AND created_at < datetime('now', '-2 months')",
              (field_id,), commit=True)

    field = query("SELECT * FROM fields WHERE id = ?", (field_id,), fetchone=True)
    if not field:
        return jsonify({"status": "error", "message": "Field not found"})

    scans = query("SELECT * FROM scans WHERE field_id = ? ORDER BY created_at DESC",
                  (field_id,), fetchall=True)

    field['status'] = 'unknown'
    if scans:
        latest = scans[0]
        if latest['result_disease'] == 'Not a Plant / Unrecognized':
            for scan in scans:
                if scan['result_disease'] != 'Not a Plant / Unrecognized':
                    field['status'] = 'healthy' if 'healthy' in (scan['result_disease'] or '').lower() else 'attention_needed'
                    break
        else:
            field['status'] = 'healthy' if 'healthy' in (latest['result_disease'] or '').lower() else 'attention_needed'

    field['recent_scans'] = scans
    field['total_scans']  = len(scans)
    return jsonify({"status": "success", "data": field})

# ─────────────────────────────────────────────────────────────────────────────
# Routes — Moisture
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/save_moisture.php', methods=['GET', 'POST'])
def save_moisture():
    field_id = request.values.get('field_id', 1, type=int)
    moisture = request.values.get('moisture', type=float)
    temperature = request.values.get('temperature', type=float)
    if moisture is None:
        return jsonify({"status": "error", "message": "moisture value is required"})

    now = datetime.utcnow().isoformat()
    if temperature is not None:
        query("UPDATE fields SET latest_moisture = ?, latest_temperature = ?, moisture_updated_at = ? WHERE id = ?",
              (moisture, temperature, now, field_id), commit=True)
        return jsonify({"status": "success", "moisture": moisture, "temperature": temperature})
    else:
        query("UPDATE fields SET latest_moisture = ?, moisture_updated_at = ? WHERE id = ?",
              (moisture, now, field_id), commit=True)
        return jsonify({"status": "success", "moisture": moisture})

# ─────────────────────────────────────────────────────────────────────────────
# Routes — Farmers
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/manage_farmers.php')
def manage_farmers():
    action = request.args.get('action', '')

    if action == 'list':
        rows = query("SELECT id, username, created_at FROM users WHERE role = 'farmer'", fetchall=True)
        return jsonify({"status": "success", "farmers": rows})

    if action == 'add':
        username = request.args.get('username', '').strip()
        password = request.args.get('password', '').strip()
        if not username or not password:
            return jsonify({"status": "error", "message": "Username and password required"})
        try:
            query("INSERT INTO users (username, password, role) VALUES (?, ?, 'farmer')",
                  (username, generate_password_hash(password)), commit=True)
            return jsonify({"status": "success", "message": "Farmer account created"})
        except Exception:
            return jsonify({"status": "error", "message": "Username might already exist"})

    if action == 'delete':
        user_id = request.args.get('id', type=int)
        if not user_id:
            return jsonify({"status": "error", "message": "ID required"})
        query("DELETE FROM users WHERE id = ? AND role = 'farmer'", (user_id,), commit=True)
        return jsonify({"status": "success", "message": "Farmer account deleted"})

    return jsonify({"status": "error", "message": "Invalid action"})

# ─────────────────────────────────────────────────────────────────────────────
# Routes — Audit Logs
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/audit_log.php')
def audit_log():
    user_id = request.args.get('user_id', type=int)
    action  = request.args.get('action', '').strip()
    if not user_id or not action:
        return jsonify({"status": "error", "message": "Missing parameters"})
    query("INSERT INTO audit_logs (user_id, action) VALUES (?, ?)", (user_id, action), commit=True)
    return jsonify({"status": "success"})


@app.route('/change_username.php', methods=['POST'])
def change_username():
    user_id = request.form.get('user_id', type=int)
    new_username = request.form.get('new_username', '').strip()
    
    if not user_id or not new_username:
        return jsonify({"status": "error", "message": "Missing parameters"})
        
    try:
        # Check if username exists
        existing = query("SELECT id FROM users WHERE username = ?", (new_username,), fetchone=True)
        if existing and existing['id'] != user_id:
            return jsonify({"status": "error", "message": "Username already taken"})
            
        query("UPDATE users SET username = ? WHERE id = ?", (new_username, user_id), commit=True)
        return jsonify({"status": "success", "new_username": new_username})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_audit_logs.php')
def get_audit_logs():
    user_id = request.args.get('user_id', type=int)
    if not user_id:
        return jsonify({"status": "error", "message": "user_id is required"})

    user = query("SELECT role FROM users WHERE id = ?", (user_id,), fetchone=True)
    if not user or user['role'] != 'owner':
        return jsonify({"status": "error", "message": "Unauthorized"})

    rows = query("""
        SELECT a.id, a.action, a.created_at, u.username
        FROM audit_logs a JOIN users u ON a.user_id = u.id
        ORDER BY a.created_at DESC LIMIT 100
    """, fetchall=True)
    return jsonify({"status": "success", "data": rows})

# ─────────────────────────────────────────────────────────────────────────────
# Routes — Image Upload + AI Scan
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
    confidence_pct = round(confidence * 100, 2) if 0 < confidence <= 1.0 else round(confidence, 2)

    # Upload to Supabase Storage
    public_url = filepath
    if DATABASE_URL:
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            safe_name = secure_filename(f"{timestamp}_{file.filename}")
            
            with open(filepath, 'rb') as f_in:
                file_bytes = f_in.read()
            
            res = requests.post(
                f"{SUPABASE_URL}/storage/v1/object/scans/{safe_name}",
                headers={
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "apikey": SUPABASE_KEY,
                    "Content-Type": file.content_type or "image/jpeg"
                },
                data=file_bytes
            )
            print(f"Supabase upload response: {res.status_code} {res.text[:200]}")
            if res.status_code in (200, 201):
                public_url = f"{SUPABASE_URL}/storage/v1/object/public/scans/{safe_name}"
                print(f"Image saved to Supabase: {public_url}")
            else:
                print(f"Supabase upload failed with status {res.status_code}")
        except Exception as e:
            print("Supabase upload failed:", e)

    query("INSERT INTO scans (field_id, image_path, result_disease, confidence, recommendation) VALUES (?, ?, ?, ?, ?)",
          (field_id, public_url, disease, confidence_pct, recommendation), commit=True)
          
    # Clean up local file so Render disk doesn't fill up
    try:
        os.remove(filepath)
    except:
        pass
    if field_status != 'unknown':
        query("UPDATE fields SET status = ? WHERE id = ?", (field_status, field_id), commit=True)

    return jsonify({"status": "success", "message": "Rover image scanned successfully",
                    "data": {"disease": disease, "confidence": confidence_pct,
                             "recommendation": recommendation,
                             "field_status": field_status, "image_url": public_url}})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "AI Model not loaded."}), 500
    if 'image' not in request.files:
        return jsonify({"error": "No image in request"}), 400

    file     = request.files['image']
    filename = secure_filename(file.filename or 'upload.jpg')
    filepath = os.path.join(TEMP_FOLDER, filename)
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
# Initialize DB and model on module load for Gunicorn
try:
    init_db()
    load_ai_assets()
except Exception as e:
    print('Startup error:', e)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Agri-Guard server on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
