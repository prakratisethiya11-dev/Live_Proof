from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import logging
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
try:
    face_detector = YOLOv5("saved_models/yolov5s-face.onnx")
    anti_spoof = AntiSpoof(weights="saved_models/AntiSpoofing_bin_1.5_128.onnx")
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    face_detector = None
    anti_spoof = None

def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(real_w, x + int(l*bbox_inc)), min(real_h, y + int(l*bbox_inc))
    img = img[y1:y2, x1:x2, :]
    return cv2.copyMakeBorder(img, y1-y, int(l*bbox_inc-y2+y), x1-x, int(l*bbox_inc)-x2+x, cv2.BORDER_CONSTANT, value=[0, 0, 0])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        if ',' in data: data = data.split(',')[1]
        img_bytes = base64.b64decode(data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None: return jsonify({"status": "error", "message": "Invalid image"}), 400

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes = face_detector([img_rgb])

        if not bboxes or len(bboxes[0]) == 0:
            return jsonify({"status": "no_face", "message": "No face detected"}), 200

        bbox = bboxes[0][0].flatten()[:4].astype(int)
        face_img = increased_crop(img_rgb, bbox)
        processed_img = anti_spoof.preprocessing(face_img)
        output = anti_spoof.ort_session.run(None, {anti_spoof.input_name: processed_img})
        
        probs = output[0][0]
        prob_real = float(probs[0])
        is_live = prob_real > request.json.get('threshold', 0.5)

        return jsonify({
            "status": "success",
            "is_live": is_live,
            "confidence": prob_real,
            "message": "Live" if is_live else "Spoof"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)