import os
import base64
import numpy as np
from flask import Flask, request, jsonify
from deepface import DeepFace
from deepface.commons import distance

app = Flask(__name__)

# متغير عالمي لتخزين المكتبة بعد تحميلها مرة واحدة
deepface_module = None
verification_module = None

def initialize_deepface():
    """
    تقوم هذه الدالة بتحميل مكتبة deepface وملحقاتها مرة واحدة فقط.
    """
    global deepface_module, verification_module
    if deepface_module is None:
        print("Initializing DeepFace for the first time...")
        from deepface import DeepFace
        from deepface.commons import distance as dst
        deepface_module = DeepFace
        verification_module = dst
        print("DeepFace initialized successfully.")

MODEL_NAME = "VGG-Face"

def base64_to_temp_path(image_base64):
    if "," in image_base64:
        image_base64 = image_base64.split(',')[1]
    image_bytes = base64.b64decode(image_base64)
    temp_path = "/tmp/temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(image_bytes)
    return temp_path

@app.route('/compare', methods=['POST'])
def compare_faces():
    # --- تم إضافة هذا الجزء المهم ---
    # طباعة رأس الطلب والجسم الخام للطلب للمساعدة في التشخيص
    print(f"Received request headers: {request.headers}")
    raw_body = request.get_data(as_text=True)
    print(f"Received raw request body: {raw_body}")
    # --- نهاية الجزء المضاف ---
    
    try:
        initialize_deepface()
        
        # محاولة قراءة البيانات من الجسم الخام
        data = request.get_json()
        if not data:
             # إذا فشل get_json، نحاول فك ترميز الجسم الخام يدويًا
             import json
             data = json.loads(raw_body)

        if 'live_image_base64' not in data or 'stored_embedding' not in data:
            return jsonify({'success': False, 'message': 'بيانات مفقودة في الطلب (live_image_base64 or stored_embedding).'}), 400

        live_img_path = base64_to_temp_path(data['live_image_base64'])
        stored_embedding = data['stored_embedding']
        
        live_embedding_objs = deepface_module.represent(
            img_path=live_img_path, 
            model_name=MODEL_NAME, 
            enforce_detection=True
        )
        live_embedding = live_embedding_objs[0]['embedding']
        
        dist = verification_module.findCosineDistance(np.array(live_embedding), np.array(stored_embedding))
        
        threshold = 0.40
        is_match = dist <= threshold
        
        return jsonify({
            'success': True,
            'is_match': bool(is_match),
            'distance': float(dist)
        })
    except Exception as e:
        error_message = str(e)
        if "Face could not be detected" in error_message:
            return jsonify({'success': False, 'message': 'لم يتم العثور على وجه في الصورة الحية.'}), 400
        return jsonify({'success': False, 'message': f"خطأ عام في الخادم: {error_message}"}), 500

# (دالة generate_embedding وبقية الكود تبقى كما هي)
@app.route('/generate', methods=['POST'])
def generate_embedding():
    try:
        initialize_deepface()
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({'success': False, 'message': 'بيانات الصورة مفقودة.'}), 400

        img_path = base64_to_temp_path(data['image_base64'])
        
        embedding_objs = deepface_module.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            enforce_detection=True
        )
        
        embedding = embedding_objs[0]['embedding']
        
        return jsonify({
            'success': True,
            'embedding': embedding
        })
    except Exception as e:
        error_message = str(e)
        if "Face could not be detected" in error_message:
            return jsonify({'success': False, 'message': 'لم يتم العثور على وجه في الصورة.'}), 400
        return jsonify({'success': False, 'message': f"خطأ عام في الخادم: {error_message}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
