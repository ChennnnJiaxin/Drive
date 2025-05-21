# -----------------------------------------------------------------------------
# Main function for the driver's fatigue level estimation algorithm
# -----------------------------------------------------------------------------
# Author: Daniel Oliveira
# https://github.com/danielsousaoliveira
# -----------------------------------------------------------------------------

import cv2
from utils import *
from detection.face import *
from detection.pose import *
from state import *
import mediapipe as mp
import time
# 需要读取数据的话需要导入两个头文件
# import csv
# import os


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """ Main function to monitor the driver's state and detect signs of drowsiness.

    This function records video using the provided camera, analyzes the frames to estimate the head pose and facial landmarks,
    and then assesses the driver's condition using a variety of facial indicators such eye openness, lip position, and head pose.
    An alert is sent if the driver exhibits indicators of sleepiness.

    """

    # Thresholds defined for driver state evaluation
    marThresh = 0.7
    marThresh2 = 0.15
    headThresh = 6
    earThresh = 0.28
    blinkThresh = 10
    gazeThresh = 5

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('video/blinking_1.mp4')

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    faceMesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)

    captureFps = cap.get(cv2.CAP_PROP_FPS)

    driverState = DriverState(marThresh, marThresh2, headThresh, earThresh, blinkThresh, gazeThresh)
    headPose = HeadPose(faceMesh)
    faceDetector = FaceDetector(faceMesh, captureFps, marThresh, marThresh2, headThresh, earThresh, blinkThresh)

    # 准备 CSV 文件用于写入每帧检测结果，不读取数据的话不要放，不换视频的话不要放
    # csv_filename = "driver_fatigue_log.csv"
    # with open(csv_filename, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([
    #         "Time (s)", "State", "EAR", "MAR", "Gaze", "Yawning",
    #         "Roll", "BaseRoll", "Pitch", "BasePitch", "Yaw", "BaseYaw"
    #     ])

    startTime = time.time()
    drowsinessCounter = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame, results = headPose.process_image(frame)
        frame = headPose.estimate_pose(frame, results, True)
        roll, pitch, yaw = headPose.calculate_angles()

        frame, sleepEyes, mar, gaze, yawning, baseR, baseP, baseY, baseG = faceDetector.evaluate_face(frame, results,
                                                                                                      roll, pitch, yaw,
                                                                                                      True)

        frame, state = driverState.eval_state(frame, sleepEyes, mar, roll, pitch, yaw, gaze, yawning, baseR, baseP,
                                              baseG)

        # 打印检测到的各类特征值，便于调试和分析
        print("=" * 60)
        print(f"State: {state}")
        print(f"  - Eye Aspect Ratio (EAR): {sleepEyes:.3f}")
        print(f"  - Mouth Aspect Ratio (MAR): {mar:.3f}")
        print(f"  - Gaze: {gaze:.3f}")
        print(f"  - Yawning: {yawning}")
        print(f"  - Head Pose Angles:")
        print(f"      Roll:  {roll:.2f}° (base: {baseR:.2f}°)")
        print(f"      Pitch: {pitch:.2f}° (base: {baseP:.2f}°)")
        print(f"      Yaw:   {yaw:.2f}° (base: {baseY:.2f}°)")
        print("=" * 60)

        # 记录当前帧的疲劳检测数据到 CSV 文件，不读取的话不要放，视频一样的话不要放
        # current_time = time.time() - startTime
        # with open(csv_filename, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([
        #         f"{current_time:.2f}", state, f"{sleepEyes:.3f}", f"{mar:.3f}", f"{gaze:.3f}", yawning,
        #         f"{roll:.2f}", f"{baseR:.2f}", f"{pitch:.2f}", f"{baseP:.2f}", f"{yaw:.2f}", f"{baseY:.2f}"
        #     ])

        # Update drowsiness counter if the driver is drowsy
        if state == "Drowsy":
            drowsinessCounter += 1

        drowsinessTime = drowsinessCounter / captureFps
        drowsy = drowsinessTime / 60

        # Reset the drowsiness counter after 1 minute (can be updated)
        if time.time() - startTime >= 60:
            drowsinessCounter = 0
            startTime = time.time()

        cv2.imshow('Driver State Monitoring', frame)

        # Alert if the driver is showing signs of drowsiness for more than the threshold
        if drowsy > 0.08:
            # This will be sent to the driver's MiBand so that he gets a vibrating alert
            print("USER IS SHOWING SIGNALS OF DROWSINESS. SENDING ALERT")

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # 计算整个过程的总疲劳时间
    totalDrowsinessMinutes = drowsinessCounter / captureFps / 60

    # 设定一个阈值（例如：如果总疲劳时间超过 10% 的时间，就认为总体疲劳）
    fatigueThresholdMinutes = (time.time() - startTime) / 60 * 0.1

    if totalDrowsinessMinutes > fatigueThresholdMinutes:
        final_result = "Driver is generally FATIGUED during the session."
    else:
        final_result = "Driver is generally ALERT during the session."

    print("\n" + "=" * 60)
    print("FINAL FATIGUE ASSESSMENT")
    print(f"- Total drowsy time: {totalDrowsinessMinutes:.2f} minutes")
    print(f"- Session duration: {(time.time() - startTime) / 60:.2f} minutes")
    print(f"- Fatigue threshold: {fatigueThresholdMinutes:.2f} minutes")
    print(f"=> RESULT: {final_result}")
    print("=" * 60)


if __name__ == "__main__":
    main()