# fatigue_detector_KNN_class.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

class FatigueDetector:
    def __init__(self):
        # 读取数据
        df = pd.read_csv("driver_fatigue_log.csv")

        # 初始化标签编码器
        self.label_encoders = {
            "State": LabelEncoder(),
            "Yawning": LabelEncoder()
        }

        # 转换标签
        df["State"] = self.label_encoders["State"].fit_transform(df["State"])
        df["Yawning"] = self.label_encoders["Yawning"].fit_transform(df["Yawning"])

        # 特征列（确认列名与数据一致）
        features = ["EAR", "MAR", "Gaze", "Yawning",
                    "Roll", "BaseRoll", "Pitch", "BasePitch", 
                    "Yaw", "BaseYaw"]  # 修正拼写错误
        X = df[features]
        y = df["State"]

        # 划分数据集
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 模型训练
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(self.x_train, self.y_train)

    def judge(self, sample_dict):
        """
        输入样本字典，预测疲劳状态
        """
        feature_order = ["EAR", "MAR", "Gaze", "Yawning",
                         "Roll", "BaseRoll", "Pitch", "BasePitch",
                         "Yaw", "BaseYaw"]  # 同步修正
        sample = [sample_dict[f] for f in feature_order]

        probas = self.model.predict_proba([sample])[0]     #返回每个类别的概率
        classes = self.model.classes_     #编码和文字对应,先是probas和classes对应

        # 获取状态编码器
        state_encoder = self.label_encoders["State"]
        prob_dict = {
            state_encoder.inverse_transform([cls])[0]: round(prob, 3)    #再把classes编码转换为数字
            for cls, prob in zip(classes, probas)     #zip将数值生成配对
        }

        pred_class = self.model.predict([sample])[0]    #判断出来的状态是编码
        state = state_encoder.inverse_transform([pred_class])[0]     #编码转文字

        return {
            "状态": state,
            "概率": prob_dict
        }
