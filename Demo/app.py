from flask import Flask, render_template, request, jsonify
import os
import subprocess
import uuid

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
            # 调用另一个Python文件处理图片
            result = subprocess.run(
                ['python', 'fatigue_detection/detector(封装)', filepath],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                return jsonify({
                    'success': True,
                    'image_url': f'/fatigue_detection/images/{filename}',
                    'result': result.stdout.strip()
                })
            else:
                return jsonify({
                    'error': 'Processing failed',
                    'details': result.stderr
                }), 500

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400


if __name__ == '__main__':
    app.run(debug=True)