# -----------------------------------------------------------------------------
# Camera Calibration Script
# -----------------------------------------------------------------------------
# Author: Daniel Oliveira
# https://github.com/danielsousaoliveira
# -----------------------------------------------------------------------------

# 该代码是用来通过拍摄棋盘格图片来检测角点，计算相机参数的，最后会保存为json文件传递给utils,来进行其他的计算

import cv2
import numpy as np
import json
import os


def array_to_list(obj):
    """ Convert a numpy array to a nested list.

    Args:
        obj (np.ndarray): The input numpy array.

    Returns:
        list: The list representation of the array.
    """

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def detect_chessboard_corners(frame, patternSize):
    """
    Detect chessboard corners in a given frame.

    Args:
        frame (np.ndarray): The input frame.
        patternSize (tuple): Tuple representing the number of internal corners per a chessboard row and column.

    Returns:
        np.ndarray or None: The array of detected corners.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, patternSize,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if not ret:
        return None

    # Refine corners to subpixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return corners


def main():
    """
    Main function to perform camera calibration using a chessboard pattern.
    """

    CHECKERBOARD = (7, 10)

    objpoints = []
    imgpoints = []

    # 3D coordinates of the chessboard corners in the object frame
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    photosFolder = 'photos'
    if not os.path.exists(photosFolder):
        os.makedirs(photosFolder)

    cap = cv2.VideoCapture(0)

    counter = 0

    print("Press space when ready! \n")

    # Loop to capture 30 calibration images, maximum
    while counter < 30:

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not ret:
            break

        corners = detect_chessboard_corners(frame, CHECKERBOARD)

        # Saves the object and image points for camera calibration,
        # when space key is pressed and corners are detected
        if corners is not None and cv2.waitKey(1) & 0xFF == ord(' '):
            objpoints.append(objp)
            imgpoints.append(corners)
            img = cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret)
            filename = os.path.join(photosFolder, f'calibration_{counter}.jpg')
            cv2.imwrite(filename, img)
            counter += 1

        cv2.imshow('Calibration', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Store results in a json file
    calibrationData = {
        "camera_matrix": mtx,
        "distortion_coefficients": dist,
        "rotation_vectors": rvecs,
        "translation_vectors": tvecs
    }

    with open('camera_calibration.json', 'w') as f:
        json.dump(calibrationData, f, default=array_to_list, indent=4)


if __name__ == "__main__":
    main()