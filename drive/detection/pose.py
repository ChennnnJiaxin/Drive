# -----------------------------------------------------------------------------
# Head Posture Estimation Class
# -----------------------------------------------------------------------------
# Author: Daniel Oliveira
# https://github.com/danielsousaoliveira
# -----------------------------------------------------------------------------

# 用于分析“人脸朝向（头部姿态）”的模块，主要包括检测人脸关键点、估算头部的旋转角度

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Head Posture Estimation Class
# -----------------------------------------------------------------------------

class HeadPose:

    def __init__(self, faceMesh, cameraMatrix=None, distCoeffs=None):

        """ Initialize the HeadPose class with the provided FaceMesh object and camera parameters.

        Args:
            faceMesh (mediapipe.FaceMesh): The Mediapipe's FaceMesh object used for facial landmark detection.
            cameraMatrix (numpy.ndarray, optional): The camera matrix for perspective transformation.
            distCoeffs (numpy.ndarray, optional): The distortion coefficients for the camera.
        """

        self.faceMesh = faceMesh
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.rvec = None
        self.tvec = None

    def process_image(self, frame):

        """ Process an input frame to detect facial landmarks and convert it to the required format.

        Args:
            frame (numpy.ndarray): The input frame.

        Returns:
            tuple: A tuple containing the processed frame and the results of facial landmark detection.
        """

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results = self.faceMesh.process(image)

        image.flags.writeable = True

        self.frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.imgH, self.imgW = self.frame.shape[:2]

        self.frame = cv2.bilateralFilter(self.frame, 4, 30, 30)

        if self.cameraMatrix is None:
            self.size = self.frame.shape
            self.focalLength = self.size[1]
            self.center = (self.size[1] / 2, self.size[0] / 2)
            self.cameraMatrix = np.array(
                [[self.focalLength, 0, self.center[0]],
                 [0, self.focalLength, self.center[1]],
                 [0, 0, 1]], dtype="double"
            )

        if self.distCoeffs is None:
            self.distCoeffs = np.zeros((4, 1))

        return self.frame, results

    def estimate_pose(self, frame, results, display=False):

        """ Estimate the head pose using the detected facial landmarks.

        Args:
            frame (numpy.ndarray): The input frame as a numpy array.
            results (mediapipe.FaceMeshResults): The results of facial landmark detection.
            display (bool, optional): Flag to indicate if the estimated pose should be displayed.

        Returns:
            numpy.ndarray: The frame with the estimated head pose.
        """

        self.frame = frame
        self.face3d = []
        self.face2d = []
        self.nose2d = (0, 0)
        self.nose3d = (0, 0)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            self.nose2d = (lm.x * self.imgW, lm.y * self.imgH)
                            self.nose3d = (lm.x * self.imgW, lm.y * self.imgH, lm.z * self.imgW)

                        x, y = int(lm.x * self.imgW), int(lm.y * self.imgH)

                        self.face2d.append([x, y])
                        self.face3d.append([x, y, lm.z])

                self.face2d = np.array(self.face2d, dtype=np.float64)
                self.face3d = np.array(self.face3d, dtype=np.float64)

                success, self.rvec, self.tvec = cv2.solvePnP(self.face3d, self.face2d, self.cameraMatrix,
                                                             self.distCoeffs)

                if success:
                    self.rvec, self.tvec = cv2.solvePnPRefineVVS(self.face3d, self.face2d, self.cameraMatrix,
                                                                 self.distCoeffs, self.rvec, self.tvec)

                if display:
                    self.calculate_angles()
                    self.frame = self.display_direction()

        return self.frame

    def calculate_angles(self):

        """ Calculate the roll, pitch, and yaw angles from the estimated rotation vector.

        Returns:
            tuple: A tuple containing the roll, pitch, and yaw angles in degrees.
        """

        rmat = cv2.Rodrigues(self.rvec)[0]

        angles = cv2.RQDecomp3x3(rmat)[0]

        self.roll = angles[0] * 360
        self.pitch = angles[1] * 360
        self.yaw = angles[2] * 360

        return self.roll, self.pitch, self.yaw

    def display_direction(self):

        """ Display the head pose direction on the driver's nose.

        Returns:
            numpy.ndarray: The frame with the head pose direction displayed.
        """

        cv2.projectPoints(self.nose3d, self.rvec, self.tvec, self.cameraMatrix, self.distCoeffs)

        p1 = (int(self.nose2d[0]), int(self.nose2d[1]))
        p2 = (int(self.nose2d[0] + self.pitch * 10), int(self.nose2d[1] - self.roll * 10))

        cv2.line(self.frame, p1, p2, (0, 0, 255), 3)
        cv2.putText(self.frame, "Roll: " + str(np.round(self.roll, 2)), (500, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(self.frame, "Pitch: " + str(np.round(self.pitch, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(self.frame, "Yaw: " + str(np.round(self.yaw, 2)), (500, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)

        return self.frame