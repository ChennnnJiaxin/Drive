# fatigue_detector.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

class FatigueDetector:
    def __init__(self):
        # 初始化训练数据
        self.train_data = [
            {"ear": 0.18, "eye_closed_frames": 25, "yawn": False, "label": "疲劳"},
            {"ear": 0.3,  "eye_closed_frames": 5,  "yawn": True,  "label": "疲劳"},
            {"ear": 0.27, "eye_closed_frames": 5,  "yawn": False, "label": "清醒"},
            {"ear": 0.35, "eye_closed_frames": 2,  "yawn": False, "label": "清醒"},
            {"ear": 0.15, "eye_closed_frames": 30, "yawn": False, "label": "疲劳"},
            {"ear": 0.28, "eye_closed_frames": 1,  "yawn": True,  "label": "疲劳"},
        ]

        self.X_train = np.array([
            [d["ear"], d["eye_closed_frames"], int(d["yawn"])] for d in self.train_data
        ])
        self.y_train = np.array([d["label"] for d in self.train_data])

        # 标签编码器
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y_train)

        # 初始化并训练模型
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(self.X_train, self.y_encoded)

    def judge(self, features: dict) -> dict:
        """
        给定特征字典，返回结构化判断结果（状态 + 概率）
        :param features: {"ear": float, "eye_closed_frames": int, "yawn": bool}
        :return: {"状态": "疲劳" or "清醒", "概率": {"清醒": 0.xx, "疲劳": 0.xx}}
        """
        ear = features.get("ear", 1.0)
        eye_closed_frames = features.get("eye_closed_frames", 0)
        yawn = features.get("yawn", False)

        input_vector = np.array([[ear, eye_closed_frames, int(yawn)]])
        prediction_encoded = self.model.predict(input_vector)[0]
        label = self.label_encoder.inverse_transform([prediction_encoded])[0]

        probas = self.model.predict_proba(input_vector)[0]
        prob_dict = dict(zip(self.label_encoder.classes_, probas))

        return {
            "状态": label,
            "概率": {k: round(v, 3) for k, v in prob_dict.items()}
        }
