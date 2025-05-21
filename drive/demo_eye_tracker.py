import cv2
from eye_tracker import extract_EAR
image=cv2.imread("images/1.jpg")
if image is None:
    print("未检测到图像")
    exit()
EAR=extract_EAR(image)
if EAR is not None:
    print(f"EAR:{EAR:.4f}")
    cv2.imshow("Image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("未检测到图片或者人脸特征")