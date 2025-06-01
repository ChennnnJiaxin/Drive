from flask import Flask, render_template, request, jsonify
import os
import uuid
import json
from fatigue_detection.test_demo import process_images  # 导入检测模块

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'fatigue_detection/images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # 生成唯一文件名
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # 直接调用process_images函数处理图片
            result = process_images(filepath)

            # 如果是单张图片的结果，直接返回
            if isinstance(result, dict):
                return jsonify({
                    'success': True,
                    'image_url': f'/fatigue_detection/images/{filename}',
                    'result': {
                        'status': result['疲劳状态'],
                        'risk_level': result['疲劳风险等级']
                    }
                })
            # 如果是多张图片的结果，返回第一个结果（根据您的test_demo.py逻辑）
            elif isinstance(result, list) and len(result) > 0:
                return jsonify({
                    'success': True,
                    'image_url': f'/fatigue_detection/images/{filename}',
                    'result': {
                        'status': result[0]['疲劳状态'],
                        'risk_level': result[0]['疲劳风险等级']
                    }
                })
            else:
                return jsonify({'error': 'No valid result returned'}), 500

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400


if __name__ == '__main__':
    app.run(debug=True)