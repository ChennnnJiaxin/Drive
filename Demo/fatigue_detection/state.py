# -----------------------------------------------------------------------------
# Algorithm to define the real time driver's state
# -----------------------------------------------------------------------------
# Author: Daniel Oliveira
# https://github.com/danielsousaoliveira
# -----------------------------------------------------------------------------

import cv2


# -----------------------------------------------------------------------------
# Driver State Definition Class
# -----------------------------------------------------------------------------

class DriverState:

    def __init__(self, marThresh, marThresh2, headThresh, earThresh, blinkThresh, gazeThresh):

        """ Initialize the DriverState class with the provided thresholds.

        Args:
            marThresh (float): Threshold for mouth aspect ratio to detect yawning.
            marThresh2 (float): Threshold for mouth aspect ratio to detect talking.
            headThresh (float): Threshold for head position estimation.
            earThresh (float): Threshold for eye aspect ratio to detect sleepy eyes.
            blinkThresh (float): Threshold for blink detection.
            gazeThresh (float): Threshold for gaze estimation.
        """

        self.headState = "Stillness"
        self.mouthState = "Closed"
        self.eyeState = "Normal"
        self.state = "Stillness"
        self.marThresh = marThresh
        self.marThresh2 = marThresh2
        self.headThresh = headThresh
        self.earThresh = earThresh
        self.blinkThresh = blinkThresh
        self.gazeThresh = gazeThresh

    def eval_mouth(self, frame, mar, yawning):

        """ Evaluate the state of the driver's mouth.

        Args:
            frame (numpy.ndarray): The input frame as a numpy array.
            mar (float): The mouth aspect ratio.
            yawning (bool): Flag indicating whether the driver is yawning.

        Returns:
            str: The state of the driver's mouth.
        """

        self.mar = mar
        self.yawning = yawning

        if self.yawning or self.mar > self.marThresh:
            self.mouthState = "Yawning"
        elif self.mar >= self.marThresh2:
            self.mouthState = "Talking"
        else:
            self.mouthState = "Closed"

        cv2.putText(frame, "MOUTH: " + str(self.mouthState), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        return self.mouthState

    def eval_eyes(self, frame, sleepyEyes):

        """ Evaluate the state of the driver's eyes.

        Args:
            frame (numpy.ndarray): The input frame as a numpy array.
            sleepyEyes (bool): Flag indicating whether the driver's eyes are sleepy.

        Returns:
            str: The state of the driver's eyes.
        """

        if sleepyEyes:
            self.eyeState = "Sleepy-eyes"
        else:
            self.eyeState = "Normal"

        cv2.putText(frame, "EYES: " + str(self.eyeState), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        return self.eyeState

    def eval_head(self, frame, roll, pitch, yaw, gaze, baseR, baseP, baseG):

        """ Evaluate the state of the driver's head.

        Args:
            frame (numpy.ndarray): The input frame as a numpy array.
            roll (float): The estimated roll angle of the head.
            pitch (float): The estimated pitch angle of the head.
            yaw (float): The estimated yaw angle of the head.
            gaze (float): The estimated gaze value.
            baseR (float): The base value of roll for comparison.
            baseP (float): The base value of pitch for comparison.
            baseG (float): The base value of gaze for comparison.

        Returns:
            str: The state of the driver's head.
        """

        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.gaze = gaze

        if self.roll < baseR - self.headThresh:
            self.headState = "Nodding"
        elif self.roll > baseR + self.headThresh or self.pitch > baseP + self.headThresh or self.pitch < baseP - self.headThresh or self.gaze > baseG + self.gazeThresh:
            self.headState = "Looking aside"
        else:
            self.headState = "Stillness"

        cv2.putText(frame, "HEAD: " + str(self.headState), (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        return self.headState

    def eval_state(self, frame, sleepyeyes, mar, roll, pitch, yaw, gaze, yawning, baseR, baseP, baseG):

        """ Evaluate the overall state of the driver.

        Args:
            frame (numpy.ndarray): The input frame.
            sleepyeyes (bool): Flag indicating whether the driver's eyes are sleepy.
            mar (float): The mouth aspect ratio.
            roll (float): The estimated roll angle of the head.
            pitch (float): The estimated pitch angle of the head.
            yaw (float): The estimated yaw angle of the head.
            gaze (float): The estimated gaze value.
            yawning (bool): Flag indicating whether the driver is yawning.
            baseR (float): The base value of roll for comparison.
            baseP (float): The base value of pitch for comparison.
            baseG (float): The base value of gaze for comparison.

        Returns:
            tuple: A tuple containing the frame with text overlay and the overall state of the driver.
        """

        self.eval_mouth(frame, mar, yawning)
        self.eval_eyes(frame, sleepyeyes)
        self.eval_head(frame, roll, pitch, yaw, gaze, baseR, baseP, baseG)

        if self.headState == "Nodding" or self.eyeState == "Sleepy-eyes" or self.mouthState == "Yawning":
            self.state = "Drowsy"
        else:
            self.state = "Stillness"

        cv2.putText(frame, "STATE: " + str(self.state), (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        return frame, self.state