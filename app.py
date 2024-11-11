from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

import os
import threading
import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import base64
import io

app = Flask(__name__)
CORS(app)

# 配置 SQLite 数据库 URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # 禁止 Flask-SQLAlchemy 发出多余的信号

# 初始化 SQLAlchemy
db = SQLAlchemy(app)

# 创建模型（Model）
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    img_face = db.Column(db.LargeBinary)  # 存储人脸照片的二进制数据
    is_sign = db.Column(db.Boolean, default=False)  # 默认签到状态为 False
    img_sign = db.Column(db.LargeBinary)  # 存储签到照片的二进制数据

    def __repr__(self):
        return f'<User {self.name}>'

# 初始化人脸识别类
class FaceRecognition:
    def __init__(self):
        self.video_capture = None
        self.is_running = False
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_this_frame = True
        self.rtsp_url = ''
        self.font_path = "./SourceHanSansCN-Regular.otf"  # 中文字体
        self.font = ImageFont.truetype(self.font_path, 40)
        self.load_known_faces()

    def load_known_faces(self):
        """加载已知人脸的编码和名字"""
        with app.app_context():
            self.known_face_encodings = []
            self.known_face_names = []
            users = User.query.all()  # 查询所有用户
            for user in users:
                image = self.decode_image(user.img_face)
                face_encoding = face_recognition.face_encodings(image)
                if face_encoding:
                    self.known_face_encodings.append(face_encoding[0])
                    self.known_face_names.append(user.name)

    def add_face(self, image_data, name):
        """添加新的面部数据"""
        try:
            image = self.decode_image(image_data)
            face_encoding = face_recognition.face_encodings(image)
            if not face_encoding:
                return {"message": "No face detected.", "status": "error"}, 200
            
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            img_face_binary = base64.b64decode(image_data)
            user = User.query.filter_by(name=name).first()
            if not user:
                # 将二进制图像存储到数据库
                user = User(name=name, img_face=img_face_binary)
                db.session.add(user)
                db.session.commit()

            self.known_face_encodings.append(face_encoding[0])
            self.known_face_names.append(name)
            return {"message": "Face added successfully.", "status": "success"}, 200
        except Exception as e:
            return {"message": f"Error: {str(e)}", "status": "error"}, 200

    def delete_face(self, name):
        """删除已知人脸"""
        if name in self.known_face_names:
            index = self.known_face_names.index(name)
            self.known_face_encodings.pop(index)
            self.known_face_names.pop(index)

            # 从数据库中删除用户
            user = User.query.filter_by(name=name).first()
            if user:
                db.session.delete(user)
                db.session.commit()

            return {"message": f"Face with name '{name}' deleted.", "status": "success"}, 200
        return {"message": f"Face with name '{name}' not found.", "status": "error"}, 200

    def decode_image(self, image_data):
        """解码base64图像字符串或二进制图像数据"""
        try:
            # 如果 image_data 是 base64 编码的字符串，则解码它
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
                image_data = base64.b64decode(image_data)
            elif isinstance(image_data, str):
                image_data = base64.b64decode(image_data)
            # 如果 image_data 已经是二进制数据，则不需要额外的处理

            image = Image.open(io.BytesIO(image_data))
            return np.array(image)
        except Exception as e:
            raise ValueError(f"Error decoding image: {e}")


    def start_video_stream(self, rtsp_url):
        """启动视频流并开始面部识别"""
        with app.app_context():  # 激活应用上下文
            self.rtsp_url = rtsp_url
            self.video_capture = cv2.VideoCapture(self.rtsp_url)
            self.is_running = True
            self.load_known_faces()

            while self.video_capture.isOpened() and self.is_running:
                ret, frame = self.video_capture.read()
                if not ret:
                    break

                if self.process_this_frame:
                    self.process_frame(frame)

                self.process_this_frame = not self.process_this_frame

            self.stop_video_stream()

    def process_frame(self, frame):
        """处理每一帧图像并更新数据库"""
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        self.face_names = []
        for face_encoding in self.face_encodings:
            print(f"Processing frame: 识别到人脸")
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                user = User.query.filter_by(name=name).first()
                if user and not user.is_sign:
                    img_sign = self.convert_frame_to_binary(frame)
                    user.img_sign = img_sign
                    user.is_sign = True
                    db.session.commit()
            self.face_names.append(name)

        self.draw_faces(frame)
    

    def convert_frame_to_binary(self, frame):
        """将视频帧转换为二进制数据"""
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()  # 返回二进制数据

    def draw_faces(self, frame):
        """在视频帧上绘制面部框架和名称"""
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            bbox = draw.textbbox((0, 0), name, font=self.font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255), font=self.font)
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        cv2.imshow('Video', frame)

    def stop_video_stream(self):
        """停止视频流并释放资源"""
        if self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()
        self.is_running = False

# 初始化 FaceRecognition 实例
face_recognition_instance = FaceRecognition()

with app.app_context():
    db.create_all()

@app.route("/addface", methods=['POST'])
def add_face():
    """API端点：添加人脸数据"""
    data = request.json
    if 'image' not in data or 'name' not in data:
        return jsonify({"message": "Image and name are required.", "status": "error"}), 200
    message, status = face_recognition_instance.add_face(data['image'], os.path.splitext(data['name'])[0])
    return jsonify(message), status

@app.route("/deleteface", methods=['DELETE'])
def delete_face():
    """API端点：删除已知人脸"""
    data = request.json
    if 'name' not in data:
        return jsonify({"message": "Name is required.", "status": "error"}), 200
    message, status = face_recognition_instance.delete_face(data['name'])
    return jsonify(message), status

@app.route("/startsign", methods=['POST'])
def start_sign():
    """API端点：开始面部识别"""
    data = request.json
    rtsp_url = data.get('rtsp_url')
    if not rtsp_url:
        return jsonify({"message": "RTSP URL is required.", "status": "error"}), 200
    if face_recognition_instance.is_running:
        return jsonify({"message": "Face recognition is already running.", "status": "error"}), 200

    thread = threading.Thread(target=face_recognition_instance.start_video_stream, args=(rtsp_url,))
    thread.daemon = True
    thread.start()
    return jsonify({"message": "Face recognition started.", "status": "success"}), 200

@app.route("/stopsign", methods=['POST'])
def stop_sign():
    """API端点：停止面部识别"""
    if not face_recognition_instance.is_running:
        return jsonify({"message": "No active face recognition session.", "status": "error"}), 200

    face_recognition_instance.stop_video_stream()
    return jsonify({"message": "Face recognition stopped.", "status": "success"}), 200

@app.route("/isrunning", methods=['GET'])
def is_running():
    """API端点：检查面部识别是否正在运行"""
    if face_recognition_instance.is_running:
        return jsonify({"message": "Face recognition is running.", "status": "success"}), 200
    return jsonify({"message": "Face recognition is not running.", "status": "error"}), 200


@app.route("/users", methods=['GET'])
def get_users():
    """Endpoint to get all users with pagination."""
    # 获取查询参数：页码 (page) 和每页数量 (per_page)
    page = request.args.get('page', 1, type=int)  # 默认从第1页开始
    per_page = request.args.get('per_page', 10, type=int)  # 默认每页显示10条

    query = db.select(User)
    # 查询数据库，支持分页
    users_query = db.paginate(query, page=page, per_page=per_page,  max_per_page=50, error_out=False)

    # 获取分页后的数据
    users = users_query.items

    # 构建返回的用户数据列表
    users_data = []
    for user in users:
        users_data.append({
            'id': user.id,
            'name': user.name,
            'is_sign': user.is_sign,
            'img_face': base64.b64encode(user.img_face).decode('utf-8') if user.img_face else None,  # 转换为Base64
            'img_sign': base64.b64encode(user.img_sign).decode('utf-8') if user.img_sign else None   # 转换为Base64
        })
        print(user.img_face)

    # 返回数据和分页信息
    return jsonify({
        'users': users_data,
        'total': users_query.total,
        'page': page,
        'per_page': per_page,
        'pages': users_query.pages
    })

@app.route("/stats", methods=['GET'])
def get_stats():
    """API端点：获取用户统计信息"""
    try:
        total_users = User.query.count()  # 获取总用户数
        signed_in_users = User.query.filter_by(is_sign=True).count()  # 获取已签到用户数
        not_signed_in_users = User.query.filter_by(is_sign=False).count()  # 获取未签到用户数
        
        # 返回统计信息
        return jsonify({
            "total_users": total_users,
            "signed_in_users": signed_in_users,
            "not_signed_in_users": not_signed_in_users,
            "status": "success"
        }), 200
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}", "status": "error"}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
