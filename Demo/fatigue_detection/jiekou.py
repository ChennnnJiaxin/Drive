# 提供提取驾驶员疲劳检测图像特征的接口函数

import cv2
from utils import *
from detection.face import *
from detection.pose import *
from state import *
import mediapipe as mp


def extract_fatigue_features(image_path):
    """
    提取单张图像中的疲劳检测特征，用于供其他模块（如算法部分）调用。

    参数:
        image_path (str): 图像文件路径

    返回:
        dict: 包含特征值的字典，如 EAR、MAR、注视角度、是否打哈欠、姿态角 等
    """
    # 参数阈值设定
    marThresh = 0.7
    marThresh2 = 0.15
    headThresh = 6
    earThresh = 0.28
    blinkThresh = 10
    gazeThresh = 5

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot read image from path: {image_path}")

    faceMesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)

    captureFps = 30  # 默认帧率
    driverState = DriverState(marThresh, marThresh2, headThresh, earThresh, blinkThresh, gazeThresh)
    headPose = HeadPose(faceMesh)
    faceDetector = FaceDetector(faceMesh, captureFps, marThresh, marThresh2, headThresh, earThresh, blinkThresh)

    frame, results = headPose.process_image(frame)
    frame = headPose.estimate_pose(frame, results, display=False)
    roll, pitch, yaw = headPose.calculate_angles()

    frame, sleepEyes, mar, gaze, yawning, baseR, baseP, baseY, baseG = faceDetector.evaluate_face(
        frame, results, roll, pitch, yaw, display=False)

    frame, state = driverState.eval_state(frame, sleepEyes, mar, roll, pitch, yaw, gaze, yawning,
                                          baseR, baseP, baseG)

    return {
        "EAR": sleepEyes,
        "MAR": mar,
        "Gaze": gaze,
        "Yawning": yawning,
        "Roll": roll,
        "Pitch": pitch,
        "Yaw": yaw,
        "BaseRoll": baseR,
        "BasePitch": baseP,
        "BaseYaw": baseY
    }
