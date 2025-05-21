# -----------------------------------------------------------------------------
# Face Tracking and Evaluation Algorithm
# -----------------------------------------------------------------------------
# Author: Daniel Oliveira
# https://github.com/danielsousaoliveira
# -----------------------------------------------------------------------------

#这是一个主要的检测人脸特征的模块：检测和跟踪眼睛、虹膜、嘴巴的关键点，计算眼睛的闭合程度，计算嘴巴的张开程度，估计头部姿态角度
#检测打哈欠、眨眼、闭眼状态，估计PERCLOS（眼睛闭合百分比），估计注视点（Gaze），视觉显示辅助功能

import cv2
import numpy as np
from numpy import linalg as LA
import time
from detection.const import *
from detection.pose import *
from drive.utils import *


# -----------------------------------------------------------------------------
# Face Tracking Class
# -----------------------------------------------------------------------------

class FaceDetector:

    def __init__(self, faceMesh, fps, marThresh, marThresh2, headThresh, earThresh, blinkThresh, cameraMatrix=None,
                 distCoeffs=None):

        """
        Initialize the FaceDetector class with counters and thresholds

        Attributes:
        faceMesh (Object): The face mesh object for landmark detection
        fps (int): Frames per second.
        marThresh (float): Threshold for mouth aspect ratio.
        marThresh2 (float): Threshold for mouth aspect ratio (used for yawn detection).
        headThresh (float): Threshold for head position estimation.
        earThresh (float): Threshold for eye aspect ratio.
        blinkThresh (float): Threshold for blink detection.
        cameraMatrix (numpy.ndarray or None): The camera matrix for camera calibration.
        distCoeffs (numpy.ndarray or None): The distortion coefficients for camera calibration.

        initialTime (float): The initial time for PERCLOS calculation.
        initialTime2 (float): The initial time for yawning rate calculation.
        blinkCounter (int): Counter for blink detection.
        yawnCounter (int): Counter for yawning detection.
        eyeCounter (int): Counter for eye closure detection.
        yawnStatus (bool): Flag indicating whether yawning is detected.
        arRoll (list): List to store roll angles.
        arPitch (list): List to store pitch angles.
        arYaw (list): List to store yaw angles.
        arGaze (list): List to store gaze scores.
        """

        self.faceMesh = faceMesh
        self.fps = fps
        self.marThresh = marThresh
        self.marThresh2 = marThresh2
        self.headThresh = headThresh
        self.earThresh = earThresh
        self.blinkThresh = blinkThresh
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

        self.initialTime = self.initialTime2 = time.time()
        self.blinkCounter = self.yawnCounter = self.eyeCounter = 0
        self.yawnStatus = False

        self.arRoll = [0]
        self.arPitch = [0]
        self.arYaw = [0]
        self.arGaze = [0]

    def detect_eyes(self, frame, results, roll, pitch, yaw, display):

        """ Detect eyes, iris and mouth in the given frame using the detected facial landmarks.

        Args:
            frame (numpy.ndarray): The input frame.
            results (Object): The results object from the face mesh detection.
            roll (float): The estimated roll angle of the head.
            pitch (float): The estimated pitch angle of the head.
            yaw (float): The estimated yaw angle of the head.
            display (bool): Flag indicating whether to display visualizations.

        Returns:
            None
        """

        self.frame = frame
        self.perclos = self.mar = self.gaze = self.ear = self.yawning = 0
        self.imgH, self.imgW = self.frame.shape[:2]
        self.arRoll = insert_sorted(self.arRoll, roll)
        self.arPitch = insert_sorted(self.arPitch, pitch)
        self.arYaw = insert_sorted(self.arYaw, yaw)

        if results.multi_face_landmarks:
            meshCoords = [(int(point.x * self.imgW), int(point.y * self.imgH)) for point in
                          results.multi_face_landmarks[0].landmark]

            leftEye = np.array([meshCoords[p] for p in Landmarks.LEFT_EYE], dtype=np.int32)
            rightEye = np.array([meshCoords[p] for p in Landmarks.RIGHT_EYE], dtype=np.int32)
            upperLips = np.array([meshCoords[p] for p in Landmarks.UPPER_LIPS], dtype=np.int32)
            lowerLips = np.array([meshCoords[p] for p in Landmarks.LOWER_LIPS], dtype=np.int32)

            (lCx, lCy), lRadius = cv2.minEnclosingCircle(np.array(meshCoords)[Landmarks.LEFT_IRIS])
            (rCx, rCy), rRadius = cv2.minEnclosingCircle(np.array(meshCoords)[Landmarks.RIGHT_IRIS])

            leftIris = [lCx, lCy, lRadius]
            rightIris = [rCx, rCy, rRadius]

            self.baseR, self.baseP, self.baseY, self.baseG = self.define_normal_position()
            self.ear = self.calculate_eye_aspect_ratio(leftEye, rightEye)
            self.perclos, self.sleepEyes = self.calculate_perclos(self.ear, roll)
            self.mar = self.calculate_mouth_aspect_ratio(upperLips, lowerLips)
            self.gaze = self.estimate_gaze(leftEye, rightEye, leftIris, rightIris)
            self.yawnRate, self.yawnStatus = self.estimate_yawning_rate(self.mar)
            self.arGaze = insert_sorted(self.arGaze, self.gaze)

            if display:
                self.frame = self.draw_eyes_mouth(self.frame, leftEye, rightEye, upperLips, lowerLips, leftIris,
                                                  rightIris)
                cv2.putText(frame, "PERCLOS " + str(self.perclos), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 2)
                cv2.putText(frame, "EAR " + str(self.ear), (500, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, "GAZE " + str(self.gaze), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, "MAR " + str(self.mar), (500, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def define_normal_position(self):

        """ Define the normal head position based on the median values of roll, pitch, and yaw angles.
            This is done so that the camera can be placed in different angles.

        Returns:
            Tuple: The median roll, pitch and yaw angles, and the gaze scores, as a tuple.
        """

        timer = time.time() - self.initialTime

        medRoll = calculate_median(self.arRoll)
        medPitch = calculate_median(self.arPitch)
        medYaw = calculate_median(self.arYaw)
        medGaze = calculate_median(self.arGaze)

        if timer >= 600:
            self.arRoll = [medRoll]
            self.arPitch = [medPitch]
            self.arYaw = [medYaw]
            self.arGaze = [medGaze]

        return medRoll, medPitch, medYaw, medGaze

    def evaluate_face(self, frame, results, roll, pitch, yaw, display=False):

        """ Evaluate the face attributes in the given frame.

        Args:
            frame (numpy.ndarray): The input frame.
            results (Object): The results object from the face mesh detection.
            roll (float): The estimated roll angle of the head.
            pitch (float): The estimated pitch angle of the head.
            yaw (float): The estimated yaw angle of the head.
            display (bool, optional): Flag indicating whether to display visualizations.

        Returns:
            tuple: A tuple containing the processed frame, sleepEyes flag, mouth aspect ratio (mar),
            gaze score, yawnStatus flag and the base values of roll, pitch, yaw, and gaze.
        """

        self.detect_eyes(frame, results, roll, pitch, yaw, display)

        return self.frame, self.sleepEyes, self.mar, self.gaze, self.yawnStatus, self.baseR, self.baseP, self.baseY, self.baseG

    def draw_eyes_mouth(self, frame, leftEye, rightEye, upperLips, lowerLips, leftIris, rightIris):

        """ Draw lines and circles around eyes and mouth on the given frame.

        Args:
            frame (numpy.ndarray): The input frame.
            leftEye (numpy.ndarray): Left eye landmark points.
            rightEye (numpy.ndarray): Right eye landmark points.
            upperLips (numpy.ndarray): Upper lip landmark points.
            lowerLips (numpy.ndarray): Lower lip landmark points.
            leftIris (list): Left iris center and radius.
            rightIris (list): Right iris center and radius.

        Returns:
            numpy.ndarray: The frame with drawn lines and circles.
        """

        centerLeft = np.array([leftIris[0], leftIris[1]], dtype=np.int32)
        centerRight = np.array([rightIris[0], rightIris[1]], dtype=np.int32)

        cv2.circle(frame, centerLeft, int(leftIris[2]), (255, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, centerRight, int(rightIris[2]), (255, 0, 0), 1, cv2.LINE_AA)
        cv2.polylines(frame, [leftEye], True, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.polylines(frame, [rightEye], True, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.polylines(frame, [upperLips], True, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.polylines(frame, [lowerLips], True, (255, 0, 0), 1, cv2.LINE_AA)

        return frame

    def calculate_eye_aspect_ratio(self, leftEye, rightEye):

        """ Calculate the average eye aspect ratio of both eyes.

        Args:
            leftEye (numpy.ndarray): Left eye landmark points.
            rightEye (numpy.ndarray): Right eye landmark points.

        Returns:
            float: The average eye aspect ratio.
        """

        earLeft = (LA.norm(leftEye[13] - leftEye[3]) + LA.norm(leftEye[11] - leftEye[5])) / (
                    2 * LA.norm(leftEye[0] - leftEye[8]))
        earRight = (LA.norm(rightEye[13] - rightEye[3]) + LA.norm(rightEye[11] - rightEye[5])) / (
                    2 * LA.norm(rightEye[0] - rightEye[8]))

        earAvg = (earLeft + earRight) / 2

        return earAvg

    def calculate_perclos(self, earAvg, roll):

        """ Calculate the PERCLOS score and check for sleepy eyes.

        Args:
            earAvg (float): The average eye aspect ratio.
            roll (float): The estimated roll angle of the head.

        Returns:
            tuple: A tuple containing the PERCLOS score and a flag indicating sleepy eyes.
        """

        if roll > self.baseR + self.headThresh or roll < self.baseR - self.headThresh:
            earAvg = earAvg * (1 + abs(roll) / 15)

        sleepyEyes = False

        timer = time.time() - self.initialTime

        if earAvg is not None and earAvg <= self.earThresh:
            self.blinkCounter += 1
            self.eyeCounter += 1

            if self.eyeCounter > self.blinkThresh:
                sleepyEyes = True
        else:
            self.eyeCounter = 0
            sleepyEyes = False

        closedTime = self.blinkCounter / self.fps
        perclosScore = closedTime / 60

        if timer >= 60:
            self.blinkCounter = 0
            self.initialTime = time.time()

        return perclosScore, sleepyEyes

    def estimate_gaze(self, leftEye, rightEye, leftIris, rightIris):

        """ Estimate the gaze based on the distance between iris centers and eye centers.

        Args:
            leftEye (numpy.ndarray): Left eye landmark points.
            rightEye (numpy.ndarray): Right eye landmark points.
            leftIris (list): Left iris center and radius.
            rightIris (list): Right iris center and radius.

        Returns:
            float: The estimated gaze value.
        """

        xLeft = (leftEye[8][0] + leftEye[0][0]) / 2
        yLeft = (leftEye[12][1] + leftEye[4][1]) / 2
        centerLeft = np.array([xLeft, yLeft], dtype=np.int32)

        xRight = (rightEye[8][0] + rightEye[0][0]) / 2
        yRight = (rightEye[12][1] + rightEye[4][1]) / 2
        centerRight = np.array([xRight, yRight], dtype=np.int32)

        lirisCenter = np.array([leftIris[0], leftIris[1]], dtype=np.int32)
        ririsCenter = np.array([rightIris[0], rightIris[1]], dtype=np.int32)

        distLeft = LA.norm(lirisCenter - centerLeft)
        distRight = LA.norm(ririsCenter - centerRight)

        gazeAvg = (distLeft + distRight) / 2

        return gazeAvg

    def calculate_mouth_aspect_ratio(self, upperLips, lowerLips):

        """ Calculate the mouth aspect ratio.

        Args:
            upperLips (numpy.ndarray): Upper lip landmark points.
            lowerLips (numpy.ndarray): Lower lip landmark points.

        Returns:
            float: The mouth aspect ratio.
        """

        marAvg = (LA.norm(upperLips[14] - lowerLips[17])
                  + LA.norm(upperLips[12] - lowerLips[14])) / (LA.norm(upperLips[0] - upperLips[8])
                                                               + LA.norm(lowerLips[12] - lowerLips[10]))

        return marAvg

    def estimate_yawning_rate(self, mar):

        """ Estimate the yawning rate based on the mouth aspect ratio.

        Args:
            mar (float): The mouth aspect ratio.

        Returns:
            tuple: A tuple containing the yawning rate and a flag indicating if the user is yawning or not.
        """

        timer = time.time() - self.initialTime2

        prevStatus = self.yawnStatus

        if mar is not None and mar > self.marThresh:
            self.yawnStatus = True
        elif mar is not None and mar < self.marThresh2:
            self.yawnStatus = False

        if prevStatus == True and self.yawnStatus == False:
            self.yawnCounter += 1

        closedTime = self.yawnCounter / self.fps
        yawnRate = closedTime / 3600

        if timer >= 3600:
            self.yawnCounter = 0
            self.initialTime2 = time.time()

        return yawnRate, self.yawnStatus