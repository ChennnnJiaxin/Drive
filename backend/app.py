from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from fatigue_detector.detector import detect_fatigue

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 限制5MB

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/detect', methods=['POST'])
def fatigue_detection():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # 调用算法检测
        result = detect_fatigue(save_path)

        return jsonify({
            'status': 'success',
            'result': result['state'],
            'confidence': result['confidence'],
            'image_url': f'/static/uploads/{filename}'
        })

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/result')
def show_result():
    return render_template('frontend'/'上传页面.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
