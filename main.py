import os
import base64
import numpy as np
from flask import Flask, request, jsonify
from deepface import DeepFace
# --- تم التعديل هنا ---
# تم تغيير مسار استيراد وحدة التحقق إلى المسار الصحيح
from deepface.commons import distance

app = Flask(__name__)

# تحديد الموديل الذي سنستخدمه
MODEL_NAME = "VGG-Face"

def base64_to_temp_path(image_base64):
    """يحول الصورة من base64 إلى مسار ملف مؤقت."""
    if "," in image_base64:
        image_base64 = image_base64.split(',')[1]
    
    image_bytes = base64.b64decode(image_base64)
    # استخدام مجلد /tmp المتاح في بيئات الخوادم
    temp_path = "/tmp/temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(image_bytes)
    return temp_path

@app.route('/generate', methods=['POST'])
def generate_embedding():
    """
    يستقبل صورة ويقوم بإنشاء بصمة الوجه (embedding).
    """
    try:
        data = request.get_json()
        if not data or 'image_base64' not in data:
            return jsonify({'success': False, 'message': 'بيانات الصورة مفقودة.'}), 400

        img_path = base64_to_temp_path(data['image_base64'])
        
        embedding_objs = DeepFace.represent(
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

@app.route('/compare', methods=['POST'])
def compare_faces():
    """
    يستقبل صورة حية وبصمة مخزنة، ويقوم بمقارنتهما.
    """
    try:
        data = request.get_json()
        if not data or 'live_image_base64' not in data or 'stored_embedding' not in data:
            return jsonify({'success': False, 'message': 'بيانات مفقودة في الطلب.'}), 400

        live_img_path = base64_to_temp_path(data['live_image_base64'])
        stored_embedding = data['stored_embedding']
        
        live_embedding_objs = DeepFace.represent(
            img_path=live_img_path, 
            model_name=MODEL_NAME, 
            enforce_detection=True
        )
        live_embedding = live_embedding_objs[0]['embedding']
        
        # --- تم التعديل هنا ---
        # تم استخدام المسار الصحيح للدالة
        dist = distance.findCosineDistance(np.array(live_embedding), np.array(stored_embedding))
        
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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
