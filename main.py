import os
import json
import base64
import face_recognition
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)


def get_embedding_from_data(image_data_base64):
    """يستخرج الـ embedding من بيانات صورة base64."""
    try:
        image_bytes = base64.b64decode(image_data_base64)
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        image_np = np.array(pil_image)

        face_locations = face_recognition.face_locations(image_np, model="hog")
        if not face_locations:
            return None, "لم يتم العثور على وجه في الصورة."

        face_encodings = face_recognition.face_encodings(image_np, known_face_locations=[face_locations[0]])
        if face_encodings:
            return face_encodings[0], None
        else:
            return None, "تم الكشف عن وجه لكن فشل استخراج السمات."
    except Exception as e:
        return None, f"خطأ في معالجة الصورة: {str(e)}"


@app.route('/compare', methods=['POST'])
def compare_faces():
    """
    يستقبل طلب POST يحتوي على صورة حية و embedding مخزن، ويقوم بمقارنتهما.
    """
    try:
        data = request.get_json()
        if not data or 'live_image_base64' not in data or 'stored_embedding_json' not in data:
            return jsonify({'success': False, 'message': 'بيانات مفقودة في الطلب.'}), 400

        live_image_base64 = data['live_image_base64']
        stored_embedding_json = data['stored_embedding_json']
        tolerance = float(data.get('tolerance', 0.55))

        # استخراج السمات من الصورة الحية
        live_embedding, error_msg = get_embedding_from_data(live_image_base64)
        if error_msg:
            return jsonify({'success': False, 'message': error_msg}), 400

        # مقارنة السمات
        stored_embedding_list = json.loads(stored_embedding_json)
        stored_embedding_np = np.array(stored_embedding_list)

        matches = face_recognition.compare_faces([stored_embedding_np], live_embedding, tolerance=tolerance)
        distance = face_recognition.face_distance([stored_embedding_np], live_embedding)

        is_match = bool(matches[0])

        return jsonify({
            'success': True,
            'is_match': is_match,
            'distance': float(distance[0])
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f"خطأ عام في الخادم: {str(e)}"}), 500


if __name__ == "__main__":
    # هذا الجزء مهم للاستضافة على Render
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
